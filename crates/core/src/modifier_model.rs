use crate::timeline_value::*;
use crate::{Color, Time, VideoSampleMethod};
use glam::{Vec2, Vec3};
use hashbrown::HashSet;
use uuid::Uuid;

pub type KeyframeSpan = Option<(Time, Time)>;

pub trait ModifierModel {
    fn display_name(&self) -> &'static str;
    fn keywords(&self) -> &'static [&'static str] {
        &[]
    }
    fn ensure_ids(&mut self, seen: &mut HashSet<Uuid>);
    fn keyframe_span(&self) -> KeyframeSpan;
    fn number(&self, _id: Uuid) -> Option<&TimelineValue<f32>> {
        None
    }
    fn number_mut(&mut self, _id: Uuid) -> Option<&mut TimelineValue<f32>> {
        None
    }
    fn number2(&self, _id: Uuid) -> Option<&TimelineValue<glam::Vec2>> {
        None
    }
    fn number2_mut(&mut self, _id: Uuid) -> Option<&mut TimelineValue<glam::Vec2>> {
        None
    }
    fn number3(&self, _id: Uuid) -> Option<&TimelineValue<glam::Vec3>> {
        None
    }
    fn number3_mut(&mut self, _id: Uuid) -> Option<&mut TimelineValue<glam::Vec3>> {
        None
    }
    fn color_mut(&mut self, _id: Uuid) -> Option<&mut TimelineValue<Color<u8>>> {
        None
    }
    fn text(&self, _id: Uuid) -> Option<&TimelineValue<String>> {
        None
    }
    fn text_mut(&mut self, _id: Uuid) -> Option<&mut TimelineValue<String>> {
        None
    }
    fn integer(&self, _id: Uuid) -> Option<&TimelineValue<u32>> {
        None
    }
    fn integer_mut(&mut self, _id: Uuid) -> Option<&mut TimelineValue<u32>> {
        None
    }
    fn sample_method(&self, _id: Uuid) -> Option<&TimelineValue<VideoSampleMethod>> {
        None
    }
    fn sample_method_mut(&mut self, _id: Uuid) -> Option<&mut TimelineValue<VideoSampleMethod>> {
        None
    }
}

impl<T: ModifierModel + ?Sized> ModifierModel for Box<T> {
    fn display_name(&self) -> &'static str {
        (**self).display_name()
    }

    fn keywords(&self) -> &'static [&'static str] {
        (**self).keywords()
    }

    fn ensure_ids(&mut self, seen: &mut HashSet<Uuid>) {
        (**self).ensure_ids(seen)
    }

    fn keyframe_span(&self) -> KeyframeSpan {
        (**self).keyframe_span()
    }

    fn number(&self, id: Uuid) -> Option<&TimelineValue<f32>> {
        (**self).number(id)
    }

    fn number_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<f32>> {
        (**self).number_mut(id)
    }

    fn number2(&self, id: Uuid) -> Option<&TimelineValue<Vec2>> {
        (**self).number2(id)
    }

    fn number2_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<Vec2>> {
        (**self).number2_mut(id)
    }

    fn number3(&self, id: Uuid) -> Option<&TimelineValue<Vec3>> {
        (**self).number3(id)
    }

    fn number3_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<Vec3>> {
        (**self).number3_mut(id)
    }

    fn color_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<Color<u8>>> {
        (**self).color_mut(id)
    }

    fn text(&self, id: Uuid) -> Option<&TimelineValue<String>> {
        (**self).text(id)
    }

    fn text_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<String>> {
        (**self).text_mut(id)
    }

    fn integer(&self, id: Uuid) -> Option<&TimelineValue<u32>> {
        (**self).integer(id)
    }

    fn integer_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<u32>> {
        (**self).integer_mut(id)
    }

    fn sample_method(&self, id: Uuid) -> Option<&TimelineValue<VideoSampleMethod>> {
        (**self).sample_method(id)
    }

    fn sample_method_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<VideoSampleMethod>> {
        (**self).sample_method_mut(id)
    }
}

pub trait TimelineValueModel<T: TimelineValueType> {
    fn timeline_value(&self, id: Uuid) -> Option<&TimelineValue<T>>;
    fn timeline_value_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<T>>;
}

macro_rules! model_access {
    ($ty:ty, $get:ident, $get_mut:ident) => {
        impl<M: ModifierModel> TimelineValueModel<$ty> for M {
            fn timeline_value(&self, id: Uuid) -> Option<&TimelineValue<$ty>> {
                self.$get(id)
            }

            fn timeline_value_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<$ty>> {
                self.$get_mut(id)
            }
        }
    };
}

model_access!(f32, number, number_mut);
model_access!(Vec2, number2, number2_mut);
model_access!(Vec3, number3, number3_mut);
model_access!(u32, integer, integer_mut);
model_access!(VideoSampleMethod, sample_method, sample_method_mut);
model_access!(String, text, text_mut);

impl<M: ModifierModel> TimelineValueModel<Color<u8>> for M {
    fn timeline_value(&self, _id: Uuid) -> Option<&TimelineValue<Color<u8>>> {
        None
    }

    fn timeline_value_mut(&mut self, id: Uuid) -> Option<&mut TimelineValue<Color<u8>>> {
        self.color_mut(id)
    }
}

pub fn ensure_unique_id(id: &mut Uuid, seen: &mut HashSet<Uuid>) {
    while id.is_nil() || !seen.insert(*id) {
        *id = Uuid::new_v4();
    }
}
pub fn ensure_timeline_value_ids<T: TimelineValueType>(
    value: &mut TimelineValue<T>,
    seen: &mut HashSet<Uuid>,
) {
    ensure_unique_id(&mut value.id, seen);
    if let Some(value) = &mut value.expression {
        ensure_unique_id(&mut value.id, seen);
    }
    if let TimelineBase::Keyframes(values) = &mut value.base {
        for value in values {
            ensure_unique_id(value.id_mut(), seen);
        }
    }
}

pub fn timeline_value_span<T: TimelineValueType>(value: &TimelineValue<T>) -> KeyframeSpan {
    match &value.base {
        TimelineBase::Const(_) => None,
        TimelineBase::Keyframes(values) => endpoint_span(values.iter().map(TimelineKeyframe::time)),
    }
}
pub fn combine(spans: impl IntoIterator<Item = KeyframeSpan>) -> KeyframeSpan {
    endpoint_span(
        spans
            .into_iter()
            .flatten()
            .flat_map(|(start, end)| [start, end]),
    )
}
fn endpoint_span(times: impl IntoIterator<Item = Time>) -> KeyframeSpan {
    let mut times = times.into_iter();
    let first = times.next()?;
    Some(times.fold((first, first), |(start, end), time| {
        (start.min(time), end.max(time))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_unique_id_replaces_nil_uuid() {
        let mut id = Uuid::nil();
        let mut seen = HashSet::new();
        
        ensure_unique_id(&mut id, &mut seen);
        
        assert!(!id.is_nil(), "nil UUID should be replaced");
        assert!(seen.contains(&id), "new UUID should be in seen set");
    }

    #[test]
    fn test_ensure_unique_id_preserves_unique_uuid() {
        let original = Uuid::new_v4();
        let mut id = original;
        let mut seen = HashSet::new();
        
        ensure_unique_id(&mut id, &mut seen);
        
        assert_eq!(id, original, "unique UUID should not be modified");
        assert!(seen.contains(&id), "UUID should be in seen set");
    }

    #[test]
    fn test_ensure_unique_id_detects_collision() {
        let uuid1 = Uuid::new_v4();
        let mut uuid2 = uuid1; // collision
        let mut seen = HashSet::new();
        seen.insert(uuid1);
        
        ensure_unique_id(&mut uuid2, &mut seen);
        
        assert_ne!(uuid2, uuid1, "colliding UUID should be replaced");
        assert!(!uuid2.is_nil(), "replacement UUID should not be nil");
        assert!(seen.contains(&uuid1), "original UUID should still be in seen");
    }

    #[test]
    fn test_timeline_value_span_const_returns_none() {
        let value = TimelineValue {
            id: Uuid::new_v4(),
            base: TimelineBase::Const(42.0f32),
            expression: None,
        };
        
        let span = timeline_value_span(&value);
        
        assert_eq!(span, None, "const timeline value should have no span");
    }

    #[test]
    fn test_timeline_value_span_empty_keyframes_returns_none() {
        let value = TimelineValue {
            id: Uuid::new_v4(),
            base: TimelineBase::Keyframes(vec![]),
            expression: None,
        };
        
        let span = timeline_value_span(&value);
        
        assert_eq!(span, None, "empty keyframes should return None");
    }

    #[test]
    fn test_combine_empty_spans_returns_none() {
        let spans: Vec<KeyframeSpan> = vec![];
        
        let result = combine(spans);
        
        assert_eq!(result, None, "combining empty spans should return None");
    }

    #[test]
    fn test_combine_all_none_spans_returns_none() {
        let spans = vec![None, None, None];
        
        let result = combine(spans);
        
        assert_eq!(result, None, "combining only None spans should return None");
    }

    #[test]
    fn test_combine_single_span() {
        use shrimply_math_core::fraction_from_integer;
        let t1 = fraction_from_integer(0);
        let t2 = fraction_from_integer(10);
        let spans = vec![Some((t1, t2))];
        
        let result = combine(spans);
        
        assert_eq!(result, Some((t1, t2)), "single span should be returned unchanged");
    }

    #[test]
    fn test_combine_multiple_overlapping_spans() {
        use shrimply_math_core::fraction_from_integer;
        let t0 = fraction_from_integer(0);
        let t5 = fraction_from_integer(5);
        let t10 = fraction_from_integer(10);
        let t15 = fraction_from_integer(15);
        
        let spans = vec![Some((t0, t10)), Some((t5, t15))];
        
        let result = combine(spans);
        
        assert_eq!(result, Some((t0, t15)), "spans should be combined to min start and max end");
    }

    #[test]
    fn test_combine_with_none_and_some_spans() {
        use shrimply_math_core::fraction_from_integer;
        let t0 = fraction_from_integer(0);
        let t10 = fraction_from_integer(10);
        let t5 = fraction_from_integer(5);
        let t15 = fraction_from_integer(15);
        
        let spans = vec![None, Some((t0, t10)), None, Some((t5, t15)), None];
        
        let result = combine(spans);
        
        assert_eq!(result, Some((t0, t15)), "None spans should be ignored");
    }

    #[test]
    fn test_endpoint_span_single_time() {
        use shrimply_math_core::fraction_from_integer;
        let t = fraction_from_integer(5);
        
        let result = endpoint_span(vec![t].into_iter());
        
        assert_eq!(result, Some((t, t)), "single time should give same start and end");
    }

    #[test]
    fn test_endpoint_span_multiple_times_unsorted() {
        use shrimply_math_core::fraction_from_integer;
        let t0 = fraction_from_integer(0);
        let t5 = fraction_from_integer(5);
        let t10 = fraction_from_integer(10);
        
        let result = endpoint_span(vec![t5, t0, t10].into_iter());
        
        assert_eq!(result, Some((t0, t10)), "should find min and max regardless of order");
    }

    // Mock struct for testing ModifierModel trait
    struct TestModifier {
        display_name_val: &'static str,
    }

    impl ModifierModel for TestModifier {
        fn display_name(&self) -> &'static str {
            self.display_name_val
        }

        fn ensure_ids(&mut self, _seen: &mut HashSet<Uuid>) {
            // no-op for test
        }

        fn keyframe_span(&self) -> KeyframeSpan {
            None
        }
    }

    #[test]
    fn test_modifier_model_default_keywords() {
        let modifier = TestModifier { display_name_val: "test" };
        
        let keywords = modifier.keywords();
        
        assert_eq!(keywords, &[], "default keywords should be empty");
    }

    #[test]
    fn test_modifier_model_default_number_accessors() {
        let modifier = TestModifier { display_name_val: "test" };
        let id = Uuid::new_v4();
        
        assert_eq!(modifier.number(id), None);
        assert_eq!(modifier.number2(id), None);
        assert_eq!(modifier.number3(id), None);
        assert_eq!(modifier.text(id), None);
        assert_eq!(modifier.integer(id), None);
        assert_eq!(modifier.sample_method(id), None);
    }

    #[test]
    fn test_box_modifier_model_delegation() {
        let modifier = TestModifier { display_name_val: "boxed_test" };
        let boxed: Box<dyn ModifierModel> = Box::new(modifier);
        
        assert_eq!(boxed.display_name(), "boxed_test");
        assert_eq!(boxed.keywords(), &[]);
        assert_eq!(boxed.keyframe_span(), None);
        
        let id = Uuid::new_v4();
        assert_eq!(boxed.number(id), None);
    }
}
