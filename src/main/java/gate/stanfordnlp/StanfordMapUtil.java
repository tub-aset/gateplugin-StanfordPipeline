package gate.stanfordnlp;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.TypesafeMap;

public class StanfordMapUtil {

	public static interface Callback {

		void handle(Class<?> annotationKeyClass, TypesafeMap annotation, TypesafeMap values) throws Exception;

	}

	public static final void traversePreOrder(Class<?> annotationKeyClass, TypesafeMap annotation, Callback callback)
			throws Exception {
		Set<TypesafeMap> alreadyVisitedElements = new IdentityHashSet<>();
		traversePreOrder(annotationKeyClass, annotation, callback, alreadyVisitedElements);
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	private static final void traversePreOrder(Class<?> annotationKeyClass, TypesafeMap annotation, Callback callback,
			Set<TypesafeMap> alreadyVisitedElements) throws Exception {

		Pair<Map<Class<?>, List<TypesafeMap>>, TypesafeMap> pair = splitAnnotationIntoTraversalAndValues(annotation);
		Map<Class<?>, List<TypesafeMap>> traversalAnnotations = pair.first();
		TypesafeMap valueAnnotations = pair.second();

		for (Entry<Class<?>, List<TypesafeMap>> entry : traversalAnnotations.entrySet()) {
			Iterator<TypesafeMap> iterator = entry.getValue().iterator();
			while (iterator.hasNext()) {
				TypesafeMap childAnnotation = iterator.next();
				if (alreadyVisitedElements.contains(childAnnotation)) {
					valueAnnotations.set((Class) entry.getKey(), childAnnotation);
					iterator.remove();
				}
			}
		}

		callback.handle(annotationKeyClass, annotation, valueAnnotations);
		alreadyVisitedElements.add(annotation);

		for (Entry<Class<?>, List<TypesafeMap>> entry : traversalAnnotations.entrySet()) {
			Class<?> childKeyClass = entry.getKey();
			for (TypesafeMap childAnnotation : entry.getValue()) {
				traversePreOrder(childKeyClass, childAnnotation, callback, alreadyVisitedElements);
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
