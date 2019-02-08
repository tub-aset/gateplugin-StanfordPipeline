package gate.stanfordnlp;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.AlternativeDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.TypesafeMap;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Factory;
import gate.FeatureMap;
import gate.stanfordnlp.StanfordMapUtil.Callback;
import gate.util.InvalidOffsetException;

public class AnnotationMapper {

	private static final String FEATURE_PARENT = "_parent";
	private static final String FEATURE_CHILDREN = "_children";

	private static final String ANNOTATIONTYPE_COREF = "Coref";
	private static final String COREF_FEATURE_ANIMACY_NAME = "animacy";
	private static final String COREF_FEATURE_MENTIONTYPE_NAME = "type";
	private static final String COREF_FEATURE_REPRESENTATIVE_NAME = "representative";
	private static final String COREF_FEATURE_ISREPRESENTATIVE_NAME = "isrepresentative";
	private static final String RELATION_COREF = "Coref";

	private static final String TREE_FEATURE_LABEL = "label";
	private static final String TREE_FEATURE_VALUE = "value";
	private static final String TREE_FEATURE_SCORE = "score";

	@SuppressWarnings("deprecation")
	private static final Class<?>[] INDIVIDUAL_ANNOTATIONS = new Class[] { TreeAnnotation.class,
			BasicDependenciesAnnotation.class, EnhancedDependenciesAnnotation.class,
			EnhancedPlusPlusDependenciesAnnotation.class, AlternativeDependenciesAnnotation.class,
			edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class,
			edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedDependenciesAnnotation.class,
			edu.stanford.nlp.coref.data.CorefChain.class,
			edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class,
			edu.stanford.nlp.dcoref.CorefChain.class,
			edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class,
			edu.stanford.nlp.coref.CorefCoreAnnotations.CorefMentionsAnnotation.class };

	public static void addGateAnnotations(AnnotationSet outputAnnotationSet, TypesafeMap annotation) throws Exception {

		StanfordMapUtil.traversePreOrder(annotation.getClass(), annotation, new Callback() {

			@Override
			public void handle(Class<?> annotationKeyClass, TypesafeMap annotation, TypesafeMap values)
					throws Exception {
				Set<Class<?>> keyClasses = annotation.keySet();
				if (keyClasses.contains(CharacterOffsetBeginAnnotation.class)
						&& keyClasses.contains(CharacterOffsetEndAnnotation.class)) {
					String annotationType = getGateName(annotationKeyClass);
					Long start = annotation.get(CharacterOffsetBeginAnnotation.class).longValue();
					Long end = annotation.get(CharacterOffsetEndAnnotation.class).longValue();

					addGateAnnotation(outputAnnotationSet, annotationType, start, end, annotation, values);
				} else {
					if (keyClasses.contains(edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class)) {
						addCorefAnnotations(outputAnnotationSet, annotation);
					}
					if (keyClasses.contains(edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class)) {
						addDcorefAnnotations(outputAnnotationSet, annotation);
					}
				}
			}

		});
	}

	private static Annotation addGateAnnotation(AnnotationSet outputAnnotationSet, String annotationType, Long start,
			Long end, TypesafeMap annotation, TypesafeMap values) throws InvalidOffsetException {
		Annotation gateAnnotation = getOrAddGateAnnotation(outputAnnotationSet, annotationType, start, end);

		for (Entry<Class<?>, Object> entry : putValuesIntoFeatures(values, gateAnnotation.getFeatures()).entrySet()) {
			Class<?> valueKeyClass = entry.getKey();
			Object value = entry.getValue();
			if (value instanceof Tree) {
				Tree tree = (Tree) value;
				Integer parentId = gateAnnotation.getId();
				String valueAnnotationType = getGateName(valueKeyClass);
				Annotation treeAnnotation = addTreeAnnotation(outputAnnotationSet, valueAnnotationType, tree, parentId);
				gateAnnotation.getFeatures().put(getGateName(valueKeyClass), treeAnnotation.getId());
			} else if (value instanceof SemanticGraph) {
				SemanticGraph graph = (SemanticGraph) value;
				Integer parentId = gateAnnotation.getId();
				String valueAnnotationType = getGateName(valueKeyClass);
				List<Integer> semanticGraphAnnotationIds = new ArrayList<>();
				for (IndexedWord root : graph.getRoots()) {
					Annotation semanticGraphAnnotation = addSemanticGraphAnnotation(outputAnnotationSet,
							valueAnnotationType, graph, root, parentId);
					semanticGraphAnnotationIds.add(semanticGraphAnnotation.getId());
				}
				gateAnnotation.getFeatures().put(getGateName(valueKeyClass), semanticGraphAnnotationIds);
			}
		}
		return gateAnnotation;
	}

	private static Map<Class<?>, Object> putValuesIntoFeatures(TypesafeMap values, FeatureMap features) {
		Map<Class<?>, Object> others = new LinkedHashMap<>();

		for (Class<?> valueKeyClass : values.keySet()) {
			@SuppressWarnings({ "unchecked", "rawtypes" })
			Object value = values.get((Class) valueKeyClass);
			if (!Util.isAssignableFromAny(valueKeyClass, INDIVIDUAL_ANNOTATIONS)) {
				features.put(getGateName(valueKeyClass), value);
			} else {
				others.put(valueKeyClass, value);
			}
		}
		return others;
	}

	private static Annotation getOrAddGateAnnotation(AnnotationSet outputAnnotationSet, String annotationType,
			Long start, Long end) throws InvalidOffsetException {
		AnnotationSet others = outputAnnotationSet.get(annotationType, start, end);
		if (others.size() == 1) {
			return others.iterator().next();
		} else {
			return addGateAnnotation(outputAnnotationSet, annotationType, start, end);
		}
	}

	private static Annotation addGateAnnotation(AnnotationSet outputAnnotationSet, String annotationType, Long start,
			Long end) throws InvalidOffsetException {
		Integer id = outputAnnotationSet.add(start, end, annotationType, Factory.newFeatureMap());
		return outputAnnotationSet.get(id);
	}

	private static void addCorefAnnotations(AnnotationSet outputAnnotationSet, TypesafeMap annotation)
			throws InvalidOffsetException {
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (edu.stanford.nlp.coref.data.CorefChain cc : annotation
				.get(edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
			List<Integer> relationIds = new ArrayList<>();
			for (edu.stanford.nlp.coref.data.CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {
				CoreMap sentence = sentences.get(mention.sentNum - 1);
				List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
				Long start = Long.valueOf(tokens.get(mention.startIndex - 1).beginPosition());
				Long end = Long.valueOf(tokens.get(mention.endIndex - 2).beginPosition());
				Annotation gateAnnotation = addGateAnnotation(outputAnnotationSet, ANNOTATIONTYPE_COREF, start, end);

				FeatureMap gateFeatures = gateAnnotation.getFeatures();
				gateFeatures.put(COREF_FEATURE_ANIMACY_NAME, mention.animacy.toString());
				gateFeatures.put(COREF_FEATURE_MENTIONTYPE_NAME, mention.mentionType.toString());
				gateFeatures.put(COREF_FEATURE_REPRESENTATIVE_NAME, cc.getRepresentativeMention().mentionSpan);
				gateFeatures.put(COREF_FEATURE_ISREPRESENTATIVE_NAME, cc.getRepresentativeMention() == mention);

				relationIds.add(gateAnnotation.getId());
			}
			if (!relationIds.isEmpty()) {
				outputAnnotationSet.getRelations().addRelation(RELATION_COREF, Util.toIntArray(relationIds));
			}
		}
	}

	private static void addDcorefAnnotations(AnnotationSet outputAnnotationSet, TypesafeMap annotation)
			throws InvalidOffsetException {
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (edu.stanford.nlp.dcoref.CorefChain cc : annotation
				.get(edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
			List<Integer> relationIds = new ArrayList<>();
			for (edu.stanford.nlp.dcoref.CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {
				CoreMap sentence = sentences.get(mention.sentNum - 1);
				List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
				Long start = Long.valueOf(tokens.get(mention.startIndex - 1).beginPosition());
				Long end = Long.valueOf(tokens.get(mention.endIndex - 2).beginPosition());
				Annotation gateAnnotation = addGateAnnotation(outputAnnotationSet, ANNOTATIONTYPE_COREF, start, end);

				FeatureMap gateFeatures = gateAnnotation.getFeatures();
				gateFeatures.put(COREF_FEATURE_ANIMACY_NAME, mention.animacy.toString());
				gateFeatures.put(COREF_FEATURE_MENTIONTYPE_NAME, mention.mentionType.toString());
				gateFeatures.put(COREF_FEATURE_REPRESENTATIVE_NAME, cc.getRepresentativeMention().mentionSpan);
				gateFeatures.put(COREF_FEATURE_ISREPRESENTATIVE_NAME, cc.getRepresentativeMention() == mention);

				relationIds.add(gateAnnotation.getId());
			}
			if (!relationIds.isEmpty()) {
				outputAnnotationSet.getRelations().addRelation(RELATION_COREF, Util.toIntArray(relationIds));
			}
		}
	}

	private static Annotation addSemanticGraphAnnotation(AnnotationSet outputAnnotationSet, String annotationType,
			SemanticGraph graph, IndexedWord word, Integer parentId) throws InvalidOffsetException {
		Annotation gateAnnotation = null;
		if (word.beginPosition() >= 0 && word.endPosition() >= 0) {
			Long start = Long.valueOf(word.beginPosition());
			Long end = Long.valueOf(word.endPosition());
			gateAnnotation = addGateAnnotation(outputAnnotationSet, annotationType, start, end);
			putValuesIntoFeatures(word, gateAnnotation.getFeatures());
		}
		Integer gateId = gateAnnotation != null ? gateAnnotation.getId() : parentId;
		List<Integer> childAnnotationIds = new ArrayList<>();
		for (IndexedWord child : graph.getChildList(word)) {
			Annotation childAnnotation = addSemanticGraphAnnotation(outputAnnotationSet, annotationType, graph, child,
					gateId);
			if (childAnnotation != null) {
				childAnnotationIds.add(childAnnotation.getId());
			}
		}
		if (gateAnnotation != null) {
			gateAnnotation.getFeatures().put(FEATURE_PARENT, parentId);
			gateAnnotation.getFeatures().put(FEATURE_CHILDREN, childAnnotationIds);
		}
		return gateAnnotation;
	}

	private static Annotation addTreeAnnotation(AnnotationSet outputAnnotationSet, String annotationType, Tree tree,
			Integer parentId) throws InvalidOffsetException {
		tree.setSpans();

		Tree startLeave = tree.getLeaves().get(tree.getSpan().getSource());
		Tree endLeave = tree.getLeaves().get(tree.getSpan().getTarget());

		Integer start = null;
		Integer end = null;
		if (startLeave.label() instanceof TypesafeMap && endLeave.label() instanceof TypesafeMap) {
			start = ((TypesafeMap) startLeave.label()).get(CharacterOffsetBeginAnnotation.class);
			end = ((TypesafeMap) endLeave.label()).get(CharacterOffsetEndAnnotation.class);
		}

		Annotation gateAnnotation = addGateAnnotation(outputAnnotationSet, annotationType, start.longValue(),
				end.longValue());
		gateAnnotation.getFeatures().put(TREE_FEATURE_LABEL, tree.label() != null ? tree.label().value() : null);
		gateAnnotation.getFeatures().put(TREE_FEATURE_VALUE, tree.value());
		gateAnnotation.getFeatures().put(TREE_FEATURE_SCORE, tree.score());

		if (tree.label() instanceof TypesafeMap) {
			putValuesIntoFeatures((TypesafeMap) tree.label(), gateAnnotation.getFeatures());
		}

		List<Integer> childAnnotationIds = new ArrayList<>();
		for (int i = 0; i < tree.children().length; i++) {
			Tree child = tree.children()[i];
			Annotation childAnnotation = addTreeAnnotation(outputAnnotationSet, annotationType, child,
					gateAnnotation.getId());
			childAnnotationIds.add(childAnnotation.getId());
		}

		gateAnnotation.getFeatures().put(FEATURE_PARENT, parentId);
		gateAnnotation.getFeatures().put(FEATURE_CHILDREN, childAnnotationIds);

		return gateAnnotation;
	}

	private static String getGateName(Class<?> clazz) {
		String key = clazz.getSimpleName();
		if (key.endsWith("Annotation")) {
			key = key.substring(0, key.length() - "Annotation".length());
		}
		return key;
	}

}
