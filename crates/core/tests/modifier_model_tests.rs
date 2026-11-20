use hashbrown::HashSet;
use shrimply_core::modifier_model::{
    KeyframeSpan, ModifierModel, combine, ensure_timeline_value_ids, ensure_unique_id,
    timeline_value_span,
};
use shrimply_core::Time;
use shrimply_core::timeline_value::{
    TimelineBase, TimelineExpression, TimelineKeyframe, TimelineValue, TimelineValueType,
};
use uuid::Uuid;

struct DummyModel;

impl ModifierModel for DummyModel {
    fn display_name(&self) -> &'static str {
        "Dummy"
    }
    fn ensure_ids(&mut self, _seen: &mut HashSet<Uuid>) {}
    fn keyframe_span(&self) -> KeyframeSpan {
        None
    }
}

#[test]
fn modifier_model_defaults_return_none_and_empty_keywords() {
    let model = DummyModel;
    let id = Uuid::new_v4();

    assert_eq!(model.display_name(), "Dummy");
    assert_eq!(model.keywords(), &[] as &[&str]);
    assert!(model.number(id).is_none());
    assert!(model.number2(id).is_none());
    assert!(model.number3(id).is_none());
    assert!(model.text(id).is_none());
    assert!(model.integer(id).is_none());
    assert!(model.sample_method(id).is_none());
}

#[test]
fn modifier_model_mut_defaults_return_none() {
    let mut model = DummyModel;
    let id = Uuid::new_v4();

    assert!(model.number_mut(id).is_none());
    assert!(model.number2_mut(id).is_none());
    assert!(model.number3_mut(id).is_none());
    assert!(model.color_mut(id).is_none());
    assert!(model.text_mut(id).is_none());
    assert!(model.integer_mut(id).is_none());
    assert!(model.sample_method_mut(id).is_none());
}

#[test]
fn box_delegates_display_name_and_keywords_to_inner_model() {
    let boxed: Box<dyn ModifierModel> = Box::new(DummyModel);

    assert_eq!(boxed.display_name(), "Dummy");
    assert_eq!(boxed.keywords(), &[] as &[&str]);
}

#[test]
fn box_delegates_ensure_ids_and_keyframe_span_to_inner_model() {
    let mut boxed: Box<dyn ModifierModel> = Box::new(DummyModel);
    let mut seen = HashSet::new();

    boxed.ensure_ids(&mut seen);
    assert!(boxed.keyframe_span().is_none());
}

#[test]
fn box_delegates_accessors_to_inner_model_defaults() {
    let mut boxed: Box<dyn ModifierModel> = Box::new(DummyModel);
    let id = Uuid::new_v4();

    assert!(boxed.number(id).is_none());
    assert!(boxed.number_mut(id).is_none());
    assert!(boxed.number2(id).is_none());
    assert!(boxed.number2_mut(id).is_none());
    assert!(boxed.number3(id).is_none());
    assert!(boxed.number3_mut(id).is_none());
    assert!(boxed.color_mut(id).is_none());
    assert!(boxed.text(id).is_none());
    assert!(boxed.text_mut(id).is_none());
    assert!(boxed.integer(id).is_none());
    assert!(boxed.integer_mut(id).is_none());
    assert!(boxed.sample_method(id).is_none());
    assert!(boxed.sample_method_mut(id).is_none());
}

#[test]
fn ensure_unique_id_replaces_nil_id() {
    let mut seen = HashSet::new();
    let mut id = Uuid::nil();

    ensure_unique_id(&mut id, &mut seen);

    assert!(!id.is_nil());
    assert!(seen.contains(&id));
}

#[test]
fn ensure_unique_id_reassigns_on_collision() {
    let mut seen = HashSet::new();
    let existing = Uuid::new_v4();
    seen.insert(existing);

    let mut duplicate = existing;
    ensure_unique_id(&mut duplicate, &mut seen);

    assert_ne!(duplicate, existing);
    assert!(!duplicate.is_nil());
    assert!(seen.contains(&duplicate));
    // The original id must remain registered alongside the new one.
    assert!(seen.contains(&existing));
}

#[test]
fn ensure_unique_id_leaves_already_unique_id_untouched() {
    let mut seen = HashSet::new();
    let original = Uuid::new_v4();
    let mut id = original;

    ensure_unique_id(&mut id, &mut seen);

    assert_eq!(id, original);
    assert!(seen.contains(&original));
}

#[test]
fn ensure_timeline_value_ids_replaces_nil_base_id() {
    let mut seen = HashSet::new();
    let mut value: TimelineValue<f32> = TimelineValue::new_const(1.0);
    value.id = Uuid::nil();

    ensure_timeline_value_ids(&mut value, &mut seen);

    assert!(!value.id.is_nil());
    assert!(seen.contains(&value.id));
}

#[test]
fn ensure_timeline_value_ids_registers_expression_id() {
    let mut seen = HashSet::new();
    let mut value: TimelineValue<f32> = TimelineValue::new_const(1.0);
    value.expression = Some(TimelineExpression {
        id: Uuid::nil(),
        enabled: true,
        source: "1 + 1".to_string(),
    });

    ensure_timeline_value_ids(&mut value, &mut seen);

    let expression_id = value.expression.as_ref().unwrap().id;
    assert!(!expression_id.is_nil());
    assert_ne!(expression_id, value.id);
    assert!(seen.contains(&value.id));
    assert!(seen.contains(&expression_id));
}

#[test]
fn ensure_timeline_value_ids_deduplicates_keyframe_ids() {
    let mut seen = HashSet::new();
    let shared_id = Uuid::new_v4();
    seen.insert(shared_id);

    let mut keyframe_a = f32::keyframe(Time::from_seconds(0), 1.0);
    keyframe_a.id = shared_id;
    let mut keyframe_b = f32::keyframe(Time::from_seconds(1), 2.0);
    keyframe_b.id = shared_id;

    let mut value: TimelineValue<f32> =
        TimelineValue::new(TimelineBase::Keyframes(vec![keyframe_a, keyframe_b]));
    value.id = Uuid::nil();

    ensure_timeline_value_ids(&mut value, &mut seen);

    let TimelineBase::Keyframes(keyframes) = &value.base else {
        panic!("expected keyframes base");
    };
    assert_ne!(keyframes[0].id(), shared_id);
    assert_ne!(keyframes[1].id(), shared_id);
    assert_ne!(keyframes[0].id(), keyframes[1].id());
    assert!(seen.contains(&keyframes[0].id()));
    assert!(seen.contains(&keyframes[1].id()));
}

#[test]
fn timeline_value_span_is_none_for_const_value() {
    let value: TimelineValue<f32> = TimelineValue::new_const(5.0);

    assert_eq!(timeline_value_span(&value), None);
}

#[test]
fn timeline_value_span_is_none_for_empty_keyframes() {
    let value: TimelineValue<f32> = TimelineValue::new(TimelineBase::Keyframes(vec![]));

    assert_eq!(timeline_value_span(&value), None);
}

#[test]
fn timeline_value_span_covers_out_of_order_keyframe_times() {
    let t0 = Time::from_seconds(0);
    let t1 = Time::from_seconds(3);
    let t5 = Time::from_seconds(5);
    let keyframes = vec![
        f32::keyframe(t5, 1.0),
        f32::keyframe(t0, 2.0),
        f32::keyframe(t1, 3.0),
    ];
    let value: TimelineValue<f32> = TimelineValue::new(TimelineBase::Keyframes(keyframes));

    assert_eq!(timeline_value_span(&value), Some((t0, t5)));
}

#[test]
fn timeline_value_span_single_keyframe_is_a_zero_length_span() {
    let t = Time::from_seconds(2);
    let value: TimelineValue<f32> =
        TimelineValue::new(TimelineBase::Keyframes(vec![f32::keyframe(t, 1.0)]));

    assert_eq!(timeline_value_span(&value), Some((t, t)));
}

#[test]
fn combine_is_none_when_all_spans_are_none() {
    assert_eq!(combine(vec![None, None]), None);
}

#[test]
fn combine_skips_none_spans_and_merges_remaining() {
    let t0 = Time::from_seconds(0);
    let t3 = Time::from_seconds(3);
    let t5 = Time::from_seconds(5);
    let t8 = Time::from_seconds(8);

    let spans = vec![None, Some((t0, t3)), None, Some((t5, t8))];

    assert_eq!(combine(spans), Some((t0, t8)));
}

#[test]
fn combine_of_empty_iterator_is_none() {
    assert_eq!(combine(Vec::<KeyframeSpan>::new()), None);
}
