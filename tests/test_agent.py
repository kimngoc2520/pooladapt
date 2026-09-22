import unittest

from src.agent.router import AgenticRouter
from src.agent.workflow import HybridRAGWorkflow


class WorkflowTests(unittest.TestCase):
    def test_fixed_workflow_populates_state(self) -> None:
        workflow = HybridRAGWorkflow(
            retrieve=lambda query: [{"id": "a"}],
            characterize=lambda candidates: candidates,
            select=lambda candidates: candidates,
            rerank=lambda query, candidates: candidates,
            generate=lambda query, context: "answer",
        )
        state = AgenticRouter(workflow).run("question")
        self.assertEqual(state.answer, "answer")
        self.assertEqual(state.final_context, [{"id": "a"}])
