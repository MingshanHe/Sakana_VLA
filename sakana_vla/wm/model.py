"""Chunk world model: the DreamerV3 RSSM (dv3/) at the decision rate with outcome heads (env: sakana_vla2, GPU).

Per step (0.5 s) it predicts, from the latent state, the outcome of the action that led there:
  rew  -> geodesic progress (m, two-hot)        con  -> no collision        clr -> min clearance (m, symlog MSE)
  vis  -> target visible                        goal -> at the goal viewpoint with the target visible
Scoring a candidate chunk = imagine its first S x 5 increments and combine the predicted outcomes.
"""
import torch
import torch.nn.functional as F

from .dv3.agent import WorldModel
from .dv3.nets import Head, TwoHot, symexp, symlog  # noqa: F401  (TwoHot re-exported for callers)
from .labels import CLR_CLIP

COLLISION_CLR = 0.12  # same threshold as the simulator


class ChunkWorldModel(WorldModel):
    def __init__(self, size="size50m", img=64, img_ch=4, vec_dim=5, act_dim=25):
        super().__init__(img_ch, vec_dim, act_dim, size, img)
        self.clr = Head(self.feat_dim, self.units, 1, 1)
        self.vis = Head(self.feat_dim, self.units, 1, 1)
        self.goal = Head(self.feat_dim, self.units, 1, 1)
        self.cfg = {"size": size, "img": img, "img_ch": img_ch, "vec_dim": vec_dim, "act_dim": act_dim}

    def loss(self, batch, twohot, state=None):
        total, m, out, last = super().loss(batch, twohot, state)  # image + vector + progress (reward) + continue + KL
        ff = self.feat(out).flatten(0, 1)
        clr_t = symlog(batch["clearance"].flatten(0, 1).float().clamp(0, CLR_CLIP))
        l_clr = (self.clr(ff)[..., 0].float() - clr_t).pow(2)
        l_vis = F.binary_cross_entropy_with_logits(self.vis(ff)[..., 0].float(), batch["target_vis"].flatten(0, 1).float(),
                                                   reduction="none")
        l_goal = F.binary_cross_entropy_with_logits(self.goal(ff)[..., 0].float(), batch["at_goal"].flatten(0, 1).float(),
                                                    reduction="none")
        extra = (l_clr + l_vis + l_goal).mean()
        m = dict(m, clr=l_clr.mean(), vis=l_vis.mean(), goal=l_goal.mean(), loss=total + extra)
        return total + extra, m, out, last

    def outcomes(self, feat, twohot):
        return {"progress": twohot.mean(self.rew(feat)), "cont": torch.sigmoid(self.con(feat)[..., 0].float()),
                "clearance": symexp(self.clr(feat)[..., 0].float()), "vis": torch.sigmoid(self.vis(feat)[..., 0].float()),
                "at_goal": torch.sigmoid(self.goal(feat)[..., 0].float())}

    @torch.no_grad()
    def rollout_outcomes(self, state, actions, twohot):
        """state: dict with leading dim M; actions (M, S, A) scaled -> aggregated outcomes over the S imagined steps."""
        progress = torch.zeros(actions.shape[0], device=actions.device)
        cont = torch.ones_like(progress)
        clr_min = torch.full_like(progress, float("inf"))
        vis_end = goal_end = None
        for s in range(actions.shape[1]):
            state = self.rssm.img_step(state, actions[:, s].to(state["deter"].dtype))
            o = self.outcomes(self.feat(state), twohot)
            progress = progress + o["progress"]
            cont = cont * o["cont"]
            clr_min = torch.minimum(clr_min, o["clearance"])
            vis_end, goal_end = o["vis"], o["at_goal"]
        return {"progress": progress, "cont": cont, "clr_min": clr_min, "vis_end": vis_end, "goal_end": goal_end}


def combine_scores(o, lam_col=10.0, lam_clr=5.0, mu_vis=0.5):
    """Scalar score from aggregated outcomes: progress, minus collision risk, minus low-clearance margin, plus target in view."""
    return (o["progress"] - lam_col * (1.0 - o["cont"]) - lam_clr * torch.clamp(COLLISION_CLR + 0.1 - o["clr_min"], min=0.0)
            + mu_vis * o["vis_end"])
