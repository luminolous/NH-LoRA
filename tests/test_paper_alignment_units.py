from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.classifier import IncrementalCosineClassifier
from src.models.lora import NHLoRALayer
from src.models.losses import growth_penalty, rank_penalty
from src.models.planner import HorizonPlanner, PlannerSignals, materialize_action
from src.models.task_state import HistoryBank, TaskStateEncoder, build_history_entry, pool_vector


class _DummyModel(nn.Module):
    def __init__(self, layer: NHLoRALayer):
        super().__init__()
        self.layers = nn.ModuleDict({"0": layer})


class PaperAlignmentUnitTests(unittest.TestCase):
    def test_materialize_action_fallback_and_candidates(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=1,
            router_topk=1,
            router_temperature=1.0,
        )
        layer.add_slot(initial_rank=1, task_id=1)
        signals = PlannerSignals(
            novelty=torch.tensor([[0.9]]),
            conflict=torch.tensor([[0.9]]),
            rank_score=torch.tensor([[0.75]]),
            rank_budget=3,
            consolidate=torch.tensor([[0.6]]),
            shared_gate=torch.tensor([[0.4]]),
        )
        materialized = materialize_action(
            action="open_new_slot",
            signals=signals,
            slot_bank=layer,
            task_embedding=torch.randn(1, 8),
            task_id=2,
            max_slots_per_block=1,
        )
        self.assertEqual(materialized["fallback_action"], "expand_rank_existing_slot")
        self.assertEqual(materialized["selected_slot"], 0)
        self.assertEqual(materialized["rank_cfg"][0], 4)
        self.assertEqual(materialized["active_slot_candidates"], [0])

    def test_planner_exposes_history_attention_separately_from_materialization(self):
        planner = HorizonPlanner(
            selected_blocks=[0],
            task_embedding_dim=16,
            history_dim=11,
            hidden_dim=12,
            layer_embedding_dim=4,
            rank_min=1,
            rank_max=4,
            tau_novelty=0.5,
            tau_conflict=0.5,
        )
        signals = planner(
            0,
            task_embedding=torch.randn(1, 16),
            history_summary=torch.randn(3, 11),
        )
        self.assertIsNotNone(signals.history_attention)
        self.assertIsNotNone(signals.history_context)
        self.assertIsNotNone(signals.planner_representation)
        self.assertEqual(tuple(signals.history_attention.shape), (1, 3))
        self.assertEqual(tuple(signals.history_context.shape), (1, 16))

    def test_history_summary_and_similarity_use_summary_and_anchor(self):
        encoder = TaskStateEncoder(feature_dim=8, grad_dim=4, embedding_dim=16)
        feature_mean = torch.arange(8, dtype=torch.float32).view(1, -1)
        feature_var = torch.ones(1, 8)
        gradient_sketch = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        pooled_mean = pool_vector(feature_mean, 2)
        pooled_var = pool_vector(feature_var, 2)
        similarity_anchor = torch.nn.functional.normalize(torch.cat([pooled_mean, pooled_var], dim=-1), dim=-1)
        summary_vector = torch.cat(
            [
                pooled_mean,
                pooled_var,
                gradient_sketch,
                torch.tensor([[0.3]]),
                torch.tensor([[0.5]]),
                torch.tensor([[0.2]]),
            ],
            dim=-1,
        )
        task_state = encoder(
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=torch.zeros(1, 1),
            entropy=torch.tensor([[0.2]]),
            similarity_anchor=similarity_anchor,
            summary_vector=summary_vector,
            class_prototypes={0: feature_mean.squeeze(0)},
        )
        history = HistoryBank()
        entry = build_history_entry(
            task_id=1,
            task_state=task_state,
            usage_summary=torch.tensor([[0.3]]),
            active_rank_summary=torch.tensor([[0.5]]),
            pooled_feature_mean=pooled_mean,
            pooled_feature_var=pooled_var,
        )
        history.append(entry)
        similarity = history.mean_similarity(task_state.summary_vector, task_state.similarity_anchor)
        self.assertEqual(entry.summary_vector.shape[-1], 11)
        self.assertGreater(float(similarity.item()), 0.0)

    def test_classifier_imprinting_wires_prototypes(self):
        classifier = IncrementalCosineClassifier(feature_dim=4, tau=1.0)
        classifier.expand(3)
        prototypes = {
            0: torch.tensor([1.0, 0.0, 0.0, 0.0]),
            2: torch.tensor([0.0, 0.0, 1.0, 0.0]),
        }
        classifier.imprint_from_prototypes(prototypes, [0, 1, 2])
        self.assertTrue(torch.allclose(classifier.weight[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5))
        self.assertTrue(torch.allclose(classifier.weight[2], torch.tensor([0.0, 0.0, 1.0, 0.0]), atol=1e-5))

    def test_rank_and_growth_penalties_use_raw_planner(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=2,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
        )
        layer.add_slot(initial_rank=2, task_id=1)
        model = _DummyModel(layer)
        strong_raw = {
            0: PlannerSignals(
                novelty=torch.tensor([[0.9]]),
                conflict=torch.tensor([[0.8]]),
                rank_score=torch.tensor([[0.7]]),
                rank_budget=3,
                consolidate=torch.tensor([[0.4]]),
                shared_gate=torch.tensor([[0.5]]),
            )
        }
        weak_raw = {
            0: PlannerSignals(
                novelty=torch.tensor([[0.1]]),
                conflict=torch.tensor([[0.1]]),
                rank_score=torch.tensor([[0.1]]),
                rank_budget=1,
                consolidate=torch.tensor([[0.4]]),
                shared_gate=torch.tensor([[0.5]]),
            )
        }
        strong_rank = rank_penalty(model, strong_raw, torch.device("cpu"))
        weak_rank = rank_penalty(model, weak_raw, torch.device("cpu"))
        strong_grow = growth_penalty(strong_raw, torch.device("cpu"))
        weak_grow = growth_penalty(weak_raw, torch.device("cpu"))
        self.assertGreater(float(strong_rank.item()), float(weak_rank.item()))
        self.assertGreater(float(strong_grow.item()), float(weak_grow.item()))


if __name__ == "__main__":
    unittest.main()
