package gate.stanfordnlp;

import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;

public class AnnotationTraversal {

	public static interface Callback {

		void handle(Class<?> keyClass, CoreMap annotation, CoreMap values) throws Exception;

	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static final void preOrder(CoreMap annotation, Callback callback) throws Exception {
		for (Class<?> keyClass : annotation.keySet()) {
			Object object = annotation.get((Class) keyClass);

			if (object instanceof CoreMap) {
				CoreMap coreMap = (CoreMap) object;
				callback.handle(keyClass, coreMap, getValueAnnotations(coreMap));
			} else {
				if (object instanceof Iterable<?>) {
					Iterable<?> iterable = (Iterable<?>) object;
					for (Object item : iterable) {
						if (item instanceof CoreMap) {
							CoreMap coreMap = (CoreMap) item;
							callback.handle(keyClass, coreMap, getValueAnnotations(coreMap));
							preOrder(coreMap, callback);
						}
					}
				}
			}
		}
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static CoreMap getValueAnnotations(CoreMap annotation) {
		CoreMap valueAnnotations = new ArrayCoreMap();
		for (Class<?> keyClass : annotation.keySet()) {
			Object object = annotation.get((Class) keyClass);
			if (object instanceof CoreMap) {
				continue;
			} else if (object instanceof Iterable<?>) {
				Iterable<?> iterable = (Iterable<?>) object;
				boolean childAnnotationsFound = false;
				for (Object item : iterable) {
					if (item instanceof CoreMap) {
						childAnnotationsFound = true;
						break;
					}
				}
				if (!childAnnotationsFound) {
					valueAnnotations.set((Class) keyClass, object);
				}
			} else {
				valueAnnotations.set((Class) keyClass, object);
			}
		}
		return valueAnnotations;
	}

}
