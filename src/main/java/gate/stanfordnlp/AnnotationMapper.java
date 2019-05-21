package gate.stanfordnlp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
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
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.BinarizedTreeAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.KBestTreesAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.TypesafeMap;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Factory;
import gate.FeatureMap;
import gate.relations.Relation;
import gate.relations.RelationSet;
import gate.stanfordnlp.StanfordMapUtil.Callback;
import gate.util.InvalidOffsetException;

public class AnnotationMapper {

	public final String FEATURE_PARENT = "_parent";
	public final String FEATURE_CHILDREN = "_children";

	public final String ANNOTATIONTYPE_COREF = "Coref";
	public final String COREF_FEATURE_MENTIONTYPE = "type";
	public final String COREF_FEATURE_NUMBER = "number";
	public final String COREF_FEATURE_GENDER = "gender";
	public final String COREF_FEATURE_ANIMACY = "animacy";
	public final String COREF_FEATURE_REPRESENTATIVE = "representative";
	public final String COREF_FEATURE_ISREPRESENTATIVE = "isrepresentative";
	public final String RELATION_COREF = "Coref";

	public final String SEMANTICGRAPH_FEATURE_WEIGHT = "weight";
	public final String SEMANTICGRAPH_FEATURE_LONGNAME = "longname";
	public final String SEMANTICGRAPH_FEATURE_SHORTNAME = "shortname";
	public final String SEMANTICGRAPH_FEATURE_SPECIFIC = "specific";

	public final String TREE_FEATURE_LABEL = "label";
	public final String TREE_FEATURE_VALUE = "value";
	public final String TREE_FEATURE_SCORE = "score";

	@SuppressWarnings("deprecation")
	private final Class<?>[] INDIVIDUAL_ANNOTATIONS = new Class[] {
			// edu.stanford.nlp.trees.TreeCoreAnnotations
			TreeAnnotation.class, BinarizedTreeAnnotation.class, KBestTreesAnnotation.class,
			// edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations
			BasicDependenciesAnnotation.class, EnhancedDependenciesAnnotation.class,
			EnhancedPlusPlusDependenciesAnnotation.class, AlternativeDependenciesAnnotation.class,
			edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class,
			edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedDependenciesAnnotation.class,
			// edu.stanford.nlp.coref.CorefCoreAnnotations
			edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class,
			edu.stanford.nlp.coref.CorefCoreAnnotations.CorefMentionsAnnotation.class,
			edu.stanford.nlp.coref.data.CorefChain.class,
			// edu.stanford.nlp.dcoref.CorefCoreAnnotations
			edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class,
			edu.stanford.nlp.dcoref.CorefChain.class };

	private final AnnotationSet annotationSet;
	private final Set<Annotation> gateAnnotations;
	private final Map<Object, Annotation> annotationMapping;

	private final RelationSet relationSet;
	private final Set<Relation> gateRelations;
	private final Map<Object, Relation> relationMapping;

	public AnnotationMapper(AnnotationSet annotationSet) {
		this.annotationSet = annotationSet;
		this.gateAnnotations = new IdentityHashSet<>();
		this.annotationMapping = new IdentityHashMap<>();

		this.relationSet = annotationSet.getRelations();
		this.gateRelations = new IdentityHashSet<>();
		this.relationMapping = new IdentityHashMap<>();
	}

	public void addGateAnnotations(TypesafeMap annotation) throws Exception {

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

					addGateAnnotation(annotationType, start, end, annotation, values);
				} else {
					if (keyClasses.contains(edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class)) {
						addCorefAnnotations(annotation);
					}
					if (keyClasses.contains(edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class)) {
						addDcorefAnnotations(annotation);
					}
				}
			}

		});
	}

	private Annotation addGateAnnotation(String annotationType, Long start, Long end, TypesafeMap annotation,
			TypesafeMap values) throws InvalidOffsetException {
		Annotation gateAnnotation = getOrAddGateAnnotation(annotationType, start, end);
		annotationMapping.put(annotation, gateAnnotation);

		for (Entry<Class<?>, Object> entry : putValuesIntoFeatures(values, gateAnnotation.getFeatures()).entrySet()) {
			Class<?> valueKeyClass = entry.getKey();
			Object value = entry.getValue();
			if (value instanceof Tree) {
				Tree tree = (Tree) value;
				Integer parentId = gateAnnotation.getId();
				String valueAnnotationType = getGateName(valueKeyClass);
				Annotation treeAnnotation = addTreeAnnotations(valueAnnotationType, tree, parentId);
				gateAnnotation.getFeatures().put(getGateName(valueKeyClass), treeAnnotation.getId());
				annotationMapping.put(value, gateAnnotation);
			} else if (value instanceof SemanticGraph) {
				SemanticGraph graph = (SemanticGraph) value;
				Integer parentId = gateAnnotation.getId();
				String valueAnnotationType = getGateName(valueKeyClass);
				List<Integer> rootAnnotationIds = new ArrayList<>();
				Map<IndexedWord, Annotation> wordMapping = new HashMap<>();
				for (IndexedWord root : graph.getRoots()) {
					Annotation semanticGraphAnnotation = addSemanticGraphWordAnnotations(valueAnnotationType, graph,
							root, parentId, wordMapping, new ArrayList<>());
					rootAnnotationIds.add(semanticGraphAnnotation.getId());
					annotationMapping.putAll(wordMapping);
				}
				for (IndexedWord source : wordMapping.keySet()) {
					for (SemanticGraphEdge edge : graph.outgoingEdgeList(source)) {
						addSemanticGraphEdgeAnnotations(valueAnnotationType, edge, wordMapping);
					}
				}

				gateAnnotation.getFeatures().put(valueAnnotationType, rootAnnotationIds);
			}
		}
		return gateAnnotation;
	}

	private Map<Class<?>, Object> putValuesIntoFeatures(TypesafeMap values, FeatureMap features) {
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

	private Annotation getOrAddGateAnnotation(String annotationType, Long start, Long end)
			throws InvalidOffsetException {
		AnnotationSet others = annotationSet.get(annotationType, start, end).getCovering(annotationType, start, end);
		if (others.size() == 1) {
			return others.iterator().next();
		} else {
			return addGateAnnotation(annotationType, start, end);
		}
	}

	private Annotation addGateAnnotation(String annotationType, Long start, Long end) throws InvalidOffsetException {
		Integer id = annotationSet.add(start, end, annotationType, Factory.newFeatureMap());
		Annotation annotation = annotationSet.get(id);
		gateAnnotations.add(annotation);
		return annotation;
	}

	private void addCorefAnnotations(TypesafeMap annotation) throws InvalidOffsetException {
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (edu.stanford.nlp.coref.data.CorefChain cc : annotation
				.get(edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
			List<Integer> relationIds = new ArrayList<>();
			for (edu.stanford.nlp.coref.data.CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {
				CoreMap sentence = sentences.get(mention.sentNum - 1);
				List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
				Long start = Long.valueOf(tokens.get(mention.startIndex - 1).beginPosition());
				Long end = Long.valueOf(tokens.get(mention.endIndex - 2).endPosition());
				Annotation gateAnnotation = addGateAnnotation(ANNOTATIONTYPE_COREF, start, end);
				annotationMapping.put(mention, gateAnnotation);

				FeatureMap gateFeatures = gateAnnotation.getFeatures();
				gateFeatures.put(COREF_FEATURE_MENTIONTYPE, mention.mentionType.toString());
				gateFeatures.put(COREF_FEATURE_NUMBER, mention.number.toString());
				gateFeatures.put(COREF_FEATURE_GENDER, mention.gender.toString());
				gateFeatures.put(COREF_FEATURE_ANIMACY, mention.animacy.toString());
				gateFeatures.put(COREF_FEATURE_REPRESENTATIVE, cc.getRepresentativeMention().mentionSpan);
				gateFeatures.put(COREF_FEATURE_ISREPRESENTATIVE, cc.getRepresentativeMention() == mention);

				relationIds.add(gateAnnotation.getId());
			}
			if (!relationIds.isEmpty()) {
				Relation gateRelation = relationSet.addRelation(RELATION_COREF, Util.toIntArray(relationIds));
				gateRelations.add(gateRelation);
				relationMapping.put(cc, gateRelation);
			}
		}
	}

	private void addDcorefAnnotations(TypesafeMap annotation) throws InvalidOffsetException {
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		for (edu.stanford.nlp.dcoref.CorefChain cc : annotation
				.get(edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
			List<Integer> relationIds = new ArrayList<>();
			for (edu.stanford.nlp.dcoref.CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {
				CoreMap sentence = sentences.get(mention.sentNum - 1);
				List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
				Long start = Long.valueOf(tokens.get(mention.startIndex - 1).beginPosition());
				Long end = Long.valueOf(tokens.get(mention.endIndex - 2).endPosition());
				Annotation gateAnnotation = addGateAnnotation(ANNOTATIONTYPE_COREF, start, end);
				annotationMapping.put(mention, gateAnnotation);

				FeatureMap gateFeatures = gateAnnotation.getFeatures();
				gateFeatures.put(COREF_FEATURE_MENTIONTYPE, mention.mentionType.toString());
				gateFeatures.put(COREF_FEATURE_NUMBER, mention.number.toString());
				gateFeatures.put(COREF_FEATURE_GENDER, mention.gender.toString());
				gateFeatures.put(COREF_FEATURE_ANIMACY, mention.animacy.toString());
				gateFeatures.put(COREF_FEATURE_REPRESENTATIVE, cc.getRepresentativeMention().mentionSpan);
				gateFeatures.put(COREF_FEATURE_ISREPRESENTATIVE, cc.getRepresentativeMention() == mention);

				relationIds.add(gateAnnotation.getId());
			}
			if (!relationIds.isEmpty()) {
				Relation gateRelation = relationSet.addRelation(RELATION_COREF, Util.toIntArray(relationIds));
				gateRelations.add(gateRelation);
				relationMapping.put(cc, gateRelation);
			}
		}
	}

	private Annotation addSemanticGraphWordAnnotations(String annotationType, SemanticGraph graph, IndexedWord word,
			Integer parentId, Map<IndexedWord, Annotation> mapping, Collection<IndexedWord> parents)
			throws InvalidOffsetException {
		Annotation gateAnnotation = null;
		if (mapping.containsKey(word)) {
			gateAnnotation = mapping.get(word);
		} else if (word.beginPosition() >= 0 && word.endPosition() >= 0) {
			Long start = Long.valueOf(word.beginPosition());
			Long end = Long.valueOf(word.endPosition());
			gateAnnotation = addGateAnnotation(annotationType, start, end);
			putValuesIntoFeatures(word, gateAnnotation.getFeatures());
			mapping.put(word, gateAnnotation);
		}
		Integer gateId = gateAnnotation != null ? gateAnnotation.getId() : parentId;
		List<Integer> childAnnotationIds = new ArrayList<>();
		parents.add(word);
		for (IndexedWord child : graph.getChildList(word)) {
			if (!parents.contains(child)) {
				Annotation childAnnotation = addSemanticGraphWordAnnotations(annotationType, graph, child, gateId,
						mapping, parents);
				if (childAnnotation != null) {
					childAnnotationIds.add(childAnnotation.getId());
				}
			}
		}
		parents.remove(word);
		if (gateAnnotation != null) {
			gateAnnotation.getFeatures().put(FEATURE_PARENT, parentId);
			gateAnnotation.getFeatures().put(FEATURE_CHILDREN, childAnnotationIds);
		}
		return gateAnnotation;
	}

	private void addSemanticGraphEdgeAnnotations(String valueAnnotationType, SemanticGraphEdge edge,
			Map<IndexedWord, Annotation> wordMapping) {
		Integer sourceId = wordMapping.get(edge.getSource()).getId();
		Integer targetId = wordMapping.get(edge.getTarget()).getId();
		double weight = edge.getWeight();

		GrammaticalRelation relation = edge.getRelation();
		Relation childRelation = null;
		do {
			String longName = relation.getLongName();
			String shortName = relation.getShortName();
			String specific = relation.getSpecific();

			Relation gateRelation = relationSet.addRelation(valueAnnotationType, sourceId, targetId);
			gateRelations.add(gateRelation);
			relationMapping.put(relation, gateRelation);

			FeatureMap relationFeatureMap = gateRelation.getFeatures();
			relationFeatureMap.put(SEMANTICGRAPH_FEATURE_WEIGHT, weight);
			relationFeatureMap.put(SEMANTICGRAPH_FEATURE_LONGNAME, longName);
			relationFeatureMap.put(SEMANTICGRAPH_FEATURE_SHORTNAME, shortName);
			relationFeatureMap.put(SEMANTICGRAPH_FEATURE_SPECIFIC, specific);

			relationFeatureMap.put(FEATURE_PARENT, null);
			relationFeatureMap.put(FEATURE_CHILDREN, Collections.emptyList());

			if (childRelation != null) {
				relationFeatureMap.put(FEATURE_CHILDREN, Util.asList(childRelation.getId()));
				childRelation.getFeatures().put(FEATURE_PARENT, gateRelation.getId());
			}

			childRelation = gateRelation;
			relation = relation.getParent();
		} while (relation != null);
	}

	private Annotation addTreeAnnotations(String annotationType, Tree tree, Integer parentId)
			throws InvalidOffsetException {
		tree.setSpans();

		Tree startLeave = tree.getLeaves().get(tree.getSpan().getSource());
		Tree endLeave = tree.getLeaves().get(tree.getSpan().getTarget());

		Integer start = null;
		Integer end = null;
		if (startLeave.label() instanceof TypesafeMap && endLeave.label() instanceof TypesafeMap) {
			start = ((TypesafeMap) startLeave.label()).get(CharacterOffsetBeginAnnotation.class);
			end = ((TypesafeMap) endLeave.label()).get(CharacterOffsetEndAnnotation.class);
		}

		Annotation gateAnnotation = addGateAnnotation(annotationType, start.longValue(), end.longValue());
		annotationMapping.put(tree, gateAnnotation);
		gateAnnotation.getFeatures().put(TREE_FEATURE_LABEL, tree.label() != null ? tree.label().value() : null);
		gateAnnotation.getFeatures().put(TREE_FEATURE_VALUE, tree.value());
		gateAnnotation.getFeatures().put(TREE_FEATURE_SCORE, tree.score());

		if (tree.label() instanceof TypesafeMap) {
			putValuesIntoFeatures((TypesafeMap) tree.label(), gateAnnotation.getFeatures());
		}

		List<Integer> childAnnotationIds = new ArrayList<>();
		for (int i = 0; i < tree.children().length; i++) {
			Tree child = tree.children()[i];
			Annotation childAnnotation = addTreeAnnotations(annotationType, child, gateAnnotation.getId());
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
