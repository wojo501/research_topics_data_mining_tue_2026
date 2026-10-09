"""DoorKey checks, launched by notebooks/doorkey_benchmark.ipynb."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import data
from src.agents import Ours
from src.doorkey import DoorKey, DoorKeyModel
from src.evaluate import evaluate
from experiments.run_doorkey import paired_summary, run_config, run_sweep


class DoorKeyTests(unittest.TestCase):
    def test_encoding_and_seed(self):
        env, same = DoorKey(5, 3), DoorKey(5, 3)
        np.testing.assert_array_equal(env.T, same.T)
        self.assertEqual(env.S, 63)
        for s in range(env.S):
            self.assertEqual(env.enc(*env.dec(s)), s)
        with self.assertRaises(ValueError):
            DoorKey(6)

    def test_prerequisites_and_persistent_unlock(self):
        env = DoorKey(5, 0)
        r, c = env.door
        before = env.enc(r, c - 1, 0)
        self.assertEqual(env.step_raw(before, 2)[0], before)
        self.assertEqual(env.step_raw(before, 5), (before, -10.0, False))
        key_state = env.enc(*env.key, 0)
        picked, reward, done = env.step_raw(key_state, 4)
        self.assertEqual(env.dec(picked)[2], 1)
        self.assertEqual((reward, done), (-1.0, False))
        closed = env.enc(r, c - 1, 1)
        self.assertEqual(env.step_raw(closed, 2)[0], closed)
        opened = env.step_raw(closed, 5)[0]
        crossed = env.step_raw(opened, 2)[0]
        self.assertEqual(env.dec(crossed), (r, c, 2))
        returned = env.step_raw(crossed, 3)[0]
        self.assertEqual(returned, opened)
        self.assertEqual(env.step_raw(opened, 5), (opened, -10.0, False))

    def test_optimal_data_completes_from_all_starts(self):
        for seed in range(3):
            env = DoorKey(5, seed)
            policy, _ = env.optimal_policy()
            for start in env.starts:
                traj = data._rollout(env, start, policy, np.random.default_rng(0), 0)
                self.assertTrue(traj[-1][4])
                self.assertIn(4, [t[1] for t in traj])
                self.assertIn(5, [t[1] for t in traj])

    def test_correct_projections_match_ground_dynamics(self):
        env = DoorKey(5, 1)
        batch = [(s, a, float(env.R[s, a]), int(env.T[s, a]), bool(env.D[s, a]))
                 for s in range(env.S) for a in range(4)]
        model = DoorKeyModel(env, batch)
        self.assertFalse(model.inconsistent)
        for s in range(env.S):
            for a in range(4):
                self.assertEqual(model._prediction(s, a), env.step_raw(s, a))
        self.assertIsNone(model.projected(0, 4))
        self.assertIsNone(model.projected(0, 5))

    def test_incorrect_projection_and_execution_memory(self):
        env = DoorKey(5, 0)
        r, c = env.door
        opened, closed = env.enc(r, c - 1, 2), env.enc(r, c - 1, 0)
        nxt, reward, done = env.step_raw(opened, 2)
        transition = (opened, 2, reward, nxt, done)
        correct = DoorKeyModel(env, [transition])
        self.assertIsNone(correct.projected(closed, 2))
        wrong = DoorKeyModel(env, [transition], "ignore_door")
        self.assertNotEqual(wrong.projected(closed, 2)[0], closed)
        agent = Ours(wrong)
        agent.observe(closed, 2, closed)
        self.assertIn((closed, 2), agent.black)
        agent.reset()
        self.assertFalse(agent.black)
        mixed = DoorKeyModel(env, [transition, (closed, 2, -1.0, closed, False)], "ignore_door")
        self.assertIsNone(mixed.projected(closed, 2))
        self.assertTrue(mixed.inconsistent)

    def test_model_does_not_read_ground_transition_tables(self):
        env = DoorKey(5, 0)
        batch = data.transitions(data.expert(env, 3, 0))
        del env.T, env.R, env.D
        model = DoorKeyModel(env, batch)
        self.assertTrue(model.obs)
        for s in range(env.S):
            model.edges(s)

    def test_complete_data_recovers_optimal_return_without_queries(self):
        env = DoorKey(5, 2)
        batch = [(s, a, float(env.R[s, a]), int(env.T[s, a]), bool(env.D[s, a]))
                 for s in range(env.S) for a in range(6)]
        agent = Ours(DoorKeyModel(env, batch))
        supervisor, _ = env.optimal_policy()
        from src.evaluate import optimal_return
        metrics = evaluate(env, agent, supervisor)
        self.assertEqual(metrics["queries_per_ep"], 0.0)
        self.assertEqual(metrics["pct_delivered"], 100.0)
        self.assertAlmostEqual(metrics["ret"], optimal_return(env, supervisor))

    def test_reproducibility_and_empty_data_fallback(self):
        env = DoorKey(5, 0)
        self.assertEqual(data.noisy_expert(env, 4, 7), data.noisy_expert(env, 4, 7))
        agent = Ours(DoorKeyModel(env, []))
        supervisor, _ = env.optimal_policy()
        metrics = evaluate(env, agent, supervisor)
        self.assertEqual(metrics["pct_delivered"], 100.0)
        self.assertEqual(metrics["pct_ep_with_query"], 100.0)

    def test_runner_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            frame = run_sweep(out, sizes=(5,), n_trajs=(3,), abstractions=("correct",),
                              agent_names=("Ours", "RECO-greedy"))
            self.assertEqual(len(frame), 2)
            self.assertTrue(frame.pct_success.equals(frame.pct_delivered))
            summary = paired_summary(frame)
            self.assertEqual(summary.iloc[0].n_pairs, 1)
            self.assertTrue(np.isnan(summary.iloc[0].ret_ci_low))
            original = out.read_bytes()
            with self.assertRaises(FileExistsError):
                run_sweep(out)
            self.assertEqual(out.read_bytes(), original)
        with self.assertRaises(ValueError):
            run_config(dataset="drivers")


if __name__ == "__main__":
    unittest.main()
