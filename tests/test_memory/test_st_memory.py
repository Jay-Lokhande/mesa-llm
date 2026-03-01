from mesa_llm.memory.st_memory import ShortTermMemory


class TestShortTermMemory:
    def test_process_step_post_without_prestep_is_noop(self, mock_agent):
        memory = ShortTermMemory(agent=mock_agent, n=5, display=False)

        memory.process_step(pre_step=False)

        assert len(memory.short_term_memory) == 0
        assert memory.step_content == {}

    def test_process_step_double_post_is_noop_after_commit(self, mock_agent):
        memory = ShortTermMemory(agent=mock_agent, n=5, display=False)
        memory.add_to_memory("observation", {"status": "ready"})

        memory.process_step(pre_step=True)
        memory.process_step(pre_step=False)

        assert len(memory.short_term_memory) == 1
        assert memory.short_term_memory[-1].step == mock_agent.model.steps

        # A second post-step call should not crash or mutate committed memory.
        memory.process_step(pre_step=False)
        assert len(memory.short_term_memory) == 1
        assert memory.short_term_memory[-1].step == mock_agent.model.steps

    def test_capacity_limit_keeps_only_recent_entries(self, mock_agent):
        memory = ShortTermMemory(agent=mock_agent, n=2, display=False)

        for step in range(3):
            mock_agent.model.steps = step
            memory.add_to_memory("observation", {"counter": step})
            memory.process_step(pre_step=True)
            memory.process_step(pre_step=False)

        assert len(memory.short_term_memory) == 2
        assert [entry.step for entry in memory.short_term_memory] == [1, 2]
