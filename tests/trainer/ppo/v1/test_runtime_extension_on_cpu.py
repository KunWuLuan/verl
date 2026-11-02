# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import pytest

from verl.trainer.runtime_extension import (
    RUNTIME_EXTENSION_API_VERSION,
    RuntimeCapability,
    RuntimeEvent,
    RuntimeExtensionContext,
    RuntimeExtensionRegistry,
    RuntimeExtensionRestartRequired,
    SafePoint,
    SafePointDecision,
)


@dataclass
class RecordingExtension:
    name: str = "recording"
    api_version: int = RUNTIME_EXTENSION_API_VERSION
    required_capabilities: tuple[RuntimeCapability, ...] = (
        RuntimeCapability.EVENTS,
        RuntimeCapability.SAFE_POINTS,
    )
    events: list[RuntimeEvent] = field(default_factory=list)
    safe_points: list[SafePoint] = field(default_factory=list)
    decision: SafePointDecision = SafePointDecision.CONTINUE

    def on_event(self, context: RuntimeExtensionContext, event: RuntimeEvent) -> None:
        assert context.api_version == RUNTIME_EXTENSION_API_VERSION
        self.events.append(event)

    def on_safe_point(self, context: RuntimeExtensionContext, safe_point: SafePoint) -> SafePointDecision:
        assert context.run_id
        self.safe_points.append(safe_point)
        return self.decision


def test_register_deliver_and_close_extension():
    registry = RuntimeExtensionRegistry(
        capabilities=(RuntimeCapability.EVENTS, RuntimeCapability.SAFE_POINTS), run_id="run-1"
    )
    extension = RecordingExtension()

    assert not registry
    registration = registry.register(extension)
    assert registry
    event = registry.publish("trainer.started", step=0)
    registry.reach(SafePoint.BEFORE_FIT)

    assert extension.events == [event]
    assert event.payload == {"step": 0}
    assert extension.safe_points == [SafePoint.BEFORE_FIT]

    registration.close()
    registration.close()
    assert not registry
    registry.publish("trainer.stopped")
    registry.reach(SafePoint.AFTER_FIT)
    assert extension.events == [event]
    assert extension.safe_points == [SafePoint.BEFORE_FIT]


def test_registration_rejects_duplicate_version_and_missing_capability():
    registry = RuntimeExtensionRegistry(capabilities=(RuntimeCapability.EVENTS,))
    extension = RecordingExtension(required_capabilities=(RuntimeCapability.EVENTS,))
    registry.register(extension)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(extension)

    incompatible = RecordingExtension(name="incompatible", api_version=99)
    with pytest.raises(ValueError, match="requires API version 99"):
        registry.register(incompatible)

    missing = RecordingExtension(name="missing", required_capabilities=(RuntimeCapability.SAFE_POINTS,))
    with pytest.raises(ValueError, match="unavailable capabilities: safe_points"):
        registry.register(missing)


def test_restart_decision_stops_at_safe_point():
    registry = RuntimeExtensionRegistry(capabilities=(RuntimeCapability.SAFE_POINTS,))
    extension = RecordingExtension(
        required_capabilities=(RuntimeCapability.SAFE_POINTS,),
        decision=SafePointDecision.RESTART_REQUIRED,
    )
    registry.register(extension)

    with pytest.raises(RuntimeExtensionRestartRequired, match="requested restart at step_boundary"):
        registry.reach(SafePoint.STEP_BOUNDARY)


def _recording_registry():
    registry = RuntimeExtensionRegistry(
        capabilities=(RuntimeCapability.EVENTS, RuntimeCapability.SAFE_POINTS), run_id="run-1"
    )
    extension = RecordingExtension()
    registry.register(extension)
    return registry, extension


def test_fit_lifecycle_completes_on_success():
    registry, extension = _recording_registry()

    assert registry.run_fit_lifecycle(lambda: "result") == "result"
    assert extension.safe_points == [SafePoint.BEFORE_FIT, SafePoint.AFTER_FIT]
    assert [event.topic for event in extension.events] == ["trainer.fit.started", "trainer.fit.finished"]


@pytest.mark.parametrize("failure_stage", ["before_fit", "fit_started", "fit"])
def test_fit_lifecycle_completes_failure_and_final_paths(monkeypatch, failure_stage):
    registry, extension = _recording_registry()
    error = RuntimeError(failure_stage)
    original_reach = registry.reach
    original_publish = registry.publish

    def reach(safe_point):
        if failure_stage == "before_fit" and safe_point is SafePoint.BEFORE_FIT:
            raise error
        return original_reach(safe_point)

    def publish(topic, **payload):
        if failure_stage == "fit_started" and topic == "trainer.fit.started":
            raise error
        return original_publish(topic, **payload)

    monkeypatch.setattr(registry, "reach", reach)
    monkeypatch.setattr(registry, "publish", publish)

    def fit():
        raise error

    with pytest.raises(RuntimeError) as raised:
        registry.run_fit_lifecycle(fit)

    assert raised.value is error
    assert extension.safe_points[-2:] == [SafePoint.TERMINAL_FAILURE, SafePoint.AFTER_FIT]
    assert [event.topic for event in extension.events][-2:] == ["trainer.fit.failed", "trainer.fit.finished"]


@pytest.mark.parametrize("failure_stage", ["failure_report", "terminal_failure", "after_fit"])
def test_fit_lifecycle_continues_after_cleanup_hook_failure(monkeypatch, failure_stage):
    registry, extension = _recording_registry()
    training_error = RuntimeError("training failed")
    hook_error = RuntimeError(f"{failure_stage} failed")
    original_reach = registry.reach
    original_publish = registry.publish

    def reach(safe_point):
        if failure_stage == "terminal_failure" and safe_point is SafePoint.TERMINAL_FAILURE:
            raise hook_error
        if failure_stage == "after_fit" and safe_point is SafePoint.AFTER_FIT:
            raise hook_error
        return original_reach(safe_point)

    def publish(topic, **payload):
        if failure_stage == "failure_report" and topic == "trainer.fit.failed":
            raise hook_error
        return original_publish(topic, **payload)

    monkeypatch.setattr(registry, "reach", reach)
    monkeypatch.setattr(registry, "publish", publish)

    def fit():
        raise training_error

    with pytest.raises(RuntimeError) as raised:
        registry.run_fit_lifecycle(fit)

    assert raised.value is hook_error
    assert extension.events[-1].topic == "trainer.fit.finished"
    if failure_stage == "failure_report":
        assert extension.safe_points[-2:] == [SafePoint.TERMINAL_FAILURE, SafePoint.AFTER_FIT]
    elif failure_stage == "terminal_failure":
        assert extension.safe_points[-1] is SafePoint.AFTER_FIT
