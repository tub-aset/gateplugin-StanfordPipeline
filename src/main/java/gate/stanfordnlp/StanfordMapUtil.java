package gate.stanfordnlp;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.TypesafeMap;

public class StanfordMapUtil {

	public static interface Callback {

		void handle(Class<?> annotationKeyClass, TypesafeMap annotation, TypesafeMap values) throws Exception;

	}

	public static final void traversePreOrder(Class<?> annotationKeyClass, TypesafeMap annotation, Callback callback)
			throws Exception {

		Pair<Map<Class<?>, List<TypesafeMap>>, TypesafeMap> pair = splitAnnotationIntoTraversalAndValues(annotation);
		Map<Class<?>, List<TypesafeMap>> traversalAnnotations = pair.first();
		TypesafeMap valueAnnotations = pair.second();

		callback.handle(annotationKeyClass, annotation, valueAnnotations);

		for (Entry<Class<?>, List<TypesafeMap>> entry : traversalAnnotations.entrySet()) {
			Class<?> childKeyClass = entry.getKey();
			for (TypesafeMap childAnnotation : entry.getValue()) {
				traversePreOrder(childKeyClass, childAnnotation, callback);
			}
		}
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static Pair<Map<Class<?>, List<TypesafeMap>>, TypesafeMap> splitAnnotationIntoTraversalAndValues(
			TypesafeMap annotation) {
		Map<Class<?>, List<TypesafeMap>> traversalAnnotations = new LinkedHashMap<>();
		TypesafeMap valueAnnotations = new ArrayCoreMap();
		for (Class<?> keyClass : annotation.keySet()) {
			Object object = annotation.get((Class) keyClass);
			if (object instanceof TypesafeMap) {
				Util.addToListMap(traversalAnnotations, keyClass, (TypesafeMap) object);
			} else if (object instanceof Iterable<?>) {
				Iterable<?> iterable = (Iterable<?>) object;
				boolean childAnnotationsFound = false;
				for (Object item : iterable) {
					if (item instanceof TypesafeMap) {
						Util.addToListMap(traversalAnnotations, keyClass, (TypesafeMap) item);
						childAnnotationsFound = true;
					}
				}
				if (!childAnnotationsFound) {
					valueAnnotations.set((Class) keyClass, object);
				}
			} else {
				valueAnnotations.set((Class) keyClass, object);
			}
		}
		return new Pair<>(traversalAnnotations, valueAnnotations);
	}

}
