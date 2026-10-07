use serde_json::Value;
use shrimply_audio_modifiers::AudioModifier;
use shrimply_core::modifier_model::ModifierModel;
use shrimply_core::timeline_value::TimelineValue;
use shrimply_inspector_core::{AudioModifierControl, InspectorRuntime};
use shrimply_project::project::Time;

use crate::item::{HeaderAction, HeaderToggle, InspectorAction, InspectorItem, InspectorListItem};
use crate::section::{ControlKind, InspectorControl, InspectorSection, NumberSpec};

pub(crate) fn items(
    modifiers: &[Value],
    runtime: InspectorRuntime,
    can_paste: bool,
) -> Vec<InspectorListItem> {
    let mut items = modifiers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let modifier: AudioModifier = serde_json::from_value(value.clone())
                .expect("audio modifier inspector value must be valid");
            modifier_item(&modifier, index, modifiers.len(), runtime).boxed()
        })
        .collect::<Vec<_>>();
    let catalog = shrimply_inspector_core::audio_modifier_catalog();
    let mut section = InspectorSection::default();
    section.add(
        InspectorControl::new(ControlKind::AudioModifierMenu, "", "")
            .value(can_paste.to_string())
            .choices(
                catalog.iter().map(|choice| choice.key.clone()).collect(),
                catalog
                    .iter()
                    .map(|choice| choice.label.to_string())
                    .collect(),
            )
            .choice_search_terms(
                catalog
                    .iter()
                    .map(|choice| choice.search_text.clone())
                    .collect(),
            ),
    );
    items.push(InspectorListItem::Flat(section));
    items
}

fn modifier_item(
    modifier: &AudioModifier,
    index: usize,
    len: usize,
    runtime: InspectorRuntime,
) -> InspectorItem {
    let mut section = InspectorSection::default();
    for control in shrimply_inspector_core::audio_modifier_controls(&modifier.effect) {
        match control {
            AudioModifierControl::Cache(cache) => {
                section.controls.extend(
                    shrimply_inspector_core::audio_cache_presentation(&cache, modifier.id)
                        .section
                        .controls,
                );
            }
            AudioModifierControl::Scalar {
                path,
                label,
                value: timeline,
                presentation,
            } => scalar(&mut section, &path, label, &timeline, runtime, presentation),
            AudioModifierControl::Boolean { path, label, value } => section.add(
                InspectorControl::new(ControlKind::Boolean, path, label).value(value.to_string()),
            ),
            AudioModifierControl::Selector {
                path,
                label,
                value,
                options,
            } => section.add(crate::selector::selector(
                path,
                label,
                &value,
                options
                    .into_iter()
                    .map(|option| (option.value.to_string(), option.label.to_string())),
            )),
            AudioModifierControl::Number {
                path,
                label,
                value,
                minimum,
                maximum,
                step,
                digits,
            } => section.add(
                InspectorControl::new(ControlKind::Number, path, label)
                    .value(value.to_string())
                    .number(NumberSpec {
                        minimum,
                        maximum,
                        drag_step: step,
                        digits,
                        ..NumberSpec::default()
                    }),
            ),
            AudioModifierControl::VoiceModel { path, label, value } => {
                let models = crate::voice_models(&value);
                section.add(
                    crate::selector::selector(
                        path,
                        label,
                        &value,
                        models
                            .values
                            .iter()
                            .map(|model| (model.clone(), model.clone())),
                    )
                    .sensitive(!models.loading && models.error.is_none())
                    .tooltip(models.error.as_deref().unwrap_or(if models.loading {
                        "Loading Pneuma models…"
                    } else {
                        ""
                    })),
                );
            }
        }
    }
    for control in &mut section.controls {
        control.target_id = Some(modifier.id);
        control.audio_modifier = true;
    }

    InspectorItem::new(
        format!("audio-modifier:{}", modifier.id),
        modifier.effect.display_name(),
        section,
    )
    .reset(InspectorAction::ResetAudioModifier {
        id: modifier.id,
        effect: serde_json::to_value(shrimply_inspector_core::default_audio_modifier_effect(
            &modifier.effect,
        ))
        .expect("default audio modifier must serialize"),
    })
    .toggle(HeaderToggle {
        active: modifier.enabled,
        tooltip: if modifier.enabled {
            "Disable modifier"
        } else {
            "Enable modifier"
        },
        activate: InspectorAction::SetAudioModifierEnabled {
            id: modifier.id,
            enabled: !modifier.enabled,
        },
    })
    .actions(actions(modifier.id, index, len))
}

fn actions(id: uuid::Uuid, index: usize, len: usize) -> Vec<HeaderAction> {
    [
        ("edit-copy-symbolic", "Copy", 0_isize, true),
        ("go-up-symbolic", "Move up", -1, index > 0),
        ("go-down-symbolic", "Move down", 1, index + 1 < len),
        ("user-trash-symbolic", "Remove", 0, true),
    ]
    .into_iter()
    .enumerate()
    .map(
        |(action, (icon, tooltip, offset, sensitive))| HeaderAction {
            icon,
            tooltip,
            sensitive,
            activate: match action {
                0 => InspectorAction::CopyAudioModifier { id },
                1 | 2 => InspectorAction::MoveAudioModifier { id, offset },
                3 => InspectorAction::RemoveAudioModifier { id },
                _ => unreachable!(),
            },
        },
    )
    .collect()
}

#[allow(clippy::too_many_arguments)]
fn scalar(
    section: &mut InspectorSection,
    path: &str,
    label: &str,
    timeline: &TimelineValue<f32>,
    runtime: InspectorRuntime,
    presentation: shrimply_inspector_core::AudioModifierScalarPresentation,
) {
    section.add(
        InspectorControl::new(ControlKind::LayeredNumber, path, label)
            .value(
                (presentation.display)(timeline.value_at(runtime.local_time.unwrap_or(Time::ZERO)))
                    .to_string(),
            )
            .number(NumberSpec {
                minimum: presentation.minimum,
                maximum: presentation.maximum,
                drag_step: presentation.drag_step,
                digits: i32::try_from(presentation.digits)
                    .expect("audio modifier scalar digits must fit i32"),
                unit: presentation.unit.unwrap_or_default(),
            })
            .store_multiplier(presentation.store_multiplier)
            .layered(path, shrimply_inspector_core::LayeredState::from(timeline))
            .timeline(
                timeline.id,
                shrimply_inspector_core::audio_modifier_scalar_graph(
                    timeline,
                    runtime,
                    presentation.display,
                ),
            ),
    );
}
