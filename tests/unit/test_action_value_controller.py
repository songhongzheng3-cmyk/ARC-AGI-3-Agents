from agents.action_value_controller import ActionValueController


def test_controller_waits_for_task_signal():
    controller = ActionValueController()
    controller.observe("ACTION1", {}, 1.0, task_signal=False)

    assert controller.choose(["ACTION1", "ACTION2"]) is None


def test_controller_activates_after_task_signal():
    controller = ActionValueController(exploration=0.0)
    controller.observe("ACTION1", {}, 0.0, task_signal=False)
    controller.observe("ACTION2", {}, 100.0, task_signal=True)

    assert controller.choose(["ACTION1", "ACTION2"]) == "ACTION2"
