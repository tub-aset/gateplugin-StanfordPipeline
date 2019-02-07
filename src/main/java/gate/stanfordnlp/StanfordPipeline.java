package gate.stanfordnlp;

import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.AlternativeDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import gate.AnnotationSet;
import gate.Factory;
import gate.FeatureMap;
import gate.Resource;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleParameter;
import gate.creole.metadata.CreoleResource;
import gate.creole.metadata.Optional;
import gate.creole.metadata.RunTime;
import gate.stanfordnlp.AnnotationTraversal.Callback;

@CreoleResource(name = "StanfordPipeline", comment = "This is a simple Stanford Pipeline")
public class StanfordPipeline extends AbstractLanguageAnalyser {
	private static final long serialVersionUID = 1L;

	private static final String ANNOTATION_COREF_NAME = "Coref";
	private static final String ANNOTATION_MENTION_FEATURE_ANIMACY_NAME = "animacy";
	private static final String ANNOTATION_MENTION_FEATURE_MENTIONTYPE_NAME = "type";
	private static final String ANNOTATION_MENTION_FEATURE_REPRESENTATIVE_NAME = "representative";
	private static final String ANNOTATION_MENTION_FEATURE_ISREPRESENTATIVE_NAME = "isrepresentative";

	private static final String RELATION_COREF_NAME = "Coref";

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

	private String annotators;
	private String properties;
	private URL propertiesFile;
	private String outputASName;

	private StanfordCoreNLP pipeline;

	@Override
	public Resource init() throws ResourceInstantiationException {
		Properties props = new Properties();
		if (propertiesFile != null) {
			try {
				URLConnection connection = propertiesFile.openConnection();
				InputStream inputStream = connection.getInputStream();
				props.load(inputStream);
				inputStream.close();
			} catch (Exception e) {
				throw new ResourceInstantiationException(e);
			}
		}
		if (properties.length() > 0) {
			String[] args = Util.stringToArgs(properties);
			props.putAll(StringUtils.argsToProperties(args));
		}
		if (annotators.length() > 0) {
			props.setProperty("annotators", annotators);
		}
		try {
			pipeline = new StanfordCoreNLP(props);
		} catch (Exception e) {
			throw new ResourceInstantiationException(e);
		}
		return this;
	}

	@Override
	public void reInit() throws ResourceInstantiationException {
		init();
	}

	@Override
	public void cleanup() {
		pipeline = null;
		super.cleanup();
	}

	@Override
	public void execute() throws ExecutionException {
		try {
			AnnotationSet outputAnnotationSet = document.getAnnotations(outputASName);

			Annotation annotation = annotateContent(document.getContent().toString());

			addGateAnnotations(outputAnnotationSet, annotation);
			addCorefAnnotations(outputAnnotationSet, annotation);
			addDcorefAnnotations(outputAnnotationSet, annotation);
		} catch (Exception e) {
			throw new ExecutionException(e);
		}
	}

	private Annotation annotateContent(String content) {
		Annotation annotation = new Annotation(content);
		pipeline.annotate(annotation);
		return annotation;
	}

	private void addGateAnnotations(AnnotationSet outputAnnotationSet, CoreMap annotation) throws Exception {

		AnnotationTraversal.preOrder(annotation, new Callback() {

			@Override
			@SuppressWarnings({ "unchecked", "rawtypes" })
			public void handle(Class<?> keyClass, CoreMap annotation, CoreMap values) throws Exception {
				if (annotation.keySet().contains(CharacterOffsetBeginAnnotation.class)
						&& annotation.keySet().contains(CharacterOffsetEndAnnotation.class)) {
					Long start = annotation.get(CharacterOffsetBeginAnnotation.class).longValue();
					Long end = annotation.get(CharacterOffsetEndAnnotation.class).longValue();
					String type = getGateName(keyClass);
					AnnotationSet others = outputAnnotationSet.get(type, start, end);
					Integer id;
					if (others.size() == 1) {
						id = others.iterator().next().getId();
					} else {
						id = outputAnnotationSet.add(start, end, type, Factory.newFeatureMap());
					}
					for (Class<?> valueKeyClass : values.keySet()) {
						if (!Util.isAssignableFromAny(valueKeyClass, INDIVIDUAL_ANNOTATIONS)) {
							Object value = values.get((Class) valueKeyClass);
							outputAnnotationSet.get(id).getFeatures().put(getGateName(valueKeyClass), value);
						}
					}
				}
			}

		});
	}

	private void addCorefAnnotations(AnnotationSet outputAnnotationSet, Annotation annotation) throws Exception {

		if (annotation.keySet().contains(edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class)) {
			List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
			for (edu.stanford.nlp.coref.data.CorefChain cc : annotation
					.get(edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
				List<Integer> relationIds = new ArrayList<>();
				for (edu.stanford.nlp.coref.data.CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {
					FeatureMap map = Factory.newFeatureMap();
					map.put(ANNOTATION_MENTION_FEATURE_ANIMACY_NAME, mention.animacy.toString());
					map.put(ANNOTATION_MENTION_FEATURE_MENTIONTYPE_NAME, mention.mentionType.toString());
					map.put(ANNOTATION_MENTION_FEATURE_REPRESENTATIVE_NAME, cc.getRepresentativeMention().mentionSpan);
					map.put(ANNOTATION_MENTION_FEATURE_ISREPRESENTATIVE_NAME, cc.getRepresentativeMention() == mention);

					CoreMap sentence = sentences.get(mention.sentNum - 1);
					List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
					CoreLabel startToken = tokens.get(mention.startIndex - 1);
					CoreLabel endToken = tokens.get(mention.endIndex - 2);
					outputAnnotationSet.add((long) startToken.beginPosition(), (long) endToken.endPosition(),
							ANNOTATION_COREF_NAME, map);
				}
				if (!relationIds.isEmpty()) {
					outputAnnotationSet.getRelations().addRelation(RELATION_COREF_NAME, Util.toIntArray(relationIds));
				}
			}
		}
	}

	private void addDcorefAnnotations(AnnotationSet outputAnnotationSet, Annotation annotation) throws Exception {

		if (annotation.keySet().contains(edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class)) {
			List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
			for (edu.stanford.nlp.dcoref.CorefChain cc : annotation
					.get(edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
				List<Integer> relationIds = new ArrayList<>();
				for (edu.stanford.nlp.dcoref.CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {
					FeatureMap map = Factory.newFeatureMap();
					map.put(ANNOTATION_MENTION_FEATURE_ANIMACY_NAME, mention.animacy.toString());
					map.put(ANNOTATION_MENTION_FEATURE_MENTIONTYPE_NAME, mention.mentionType.toString());
					map.put(ANNOTATION_MENTION_FEATURE_REPRESENTATIVE_NAME, cc.getRepresentativeMention().mentionSpan);
					map.put(ANNOTATION_MENTION_FEATURE_ISREPRESENTATIVE_NAME, cc.getRepresentativeMention() == mention);

					CoreMap sentence = sentences.get(mention.sentNum - 1);
					List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
					CoreLabel startToken = tokens.get(mention.startIndex - 1);
					CoreLabel endToken = tokens.get(mention.endIndex - 2);
					outputAnnotationSet.add((long) startToken.beginPosition(), (long) endToken.endPosition(),
							ANNOTATION_COREF_NAME, map);
				}
				if (!relationIds.isEmpty()) {
					outputAnnotationSet.getRelations().addRelation(RELATION_COREF_NAME, Util.toIntArray(relationIds));
				}
			}
		}
	}

	private static String getGateName(Class<?> clazz) {
		String key = clazz.getSimpleName();
		if (key.endsWith("Annotation")) {
			key = key.substring(0, key.length() - "Annotation".length());
		}
		return key;
	}

	@CreoleParameter(comment = "StanfordNLP pipeline annotators (overrides annotators property from properties and propertiesFile)", defaultValue = "tokenize,ssplit,pos,lemma,ner,parse,mention,coref")
	public void setAnnotators(String annotators) {
		this.annotators = annotators;
	}

	public String getAnnotators() {
		return annotators;
	}

	@CreoleParameter(comment = "StanfordNLP pipeline properties (command-line style, e.g. -ner.useSUTime 0)", defaultValue = "")
	public void setProperties(String properties) {
		this.properties = properties;
	}

	public String getProperties() {
		return properties;
	}

	@CreoleParameter(comment = "StanfordNLP pipeline properties file (these properties are overridden by properties command-line string)")
	public void setPropertiesFile(URL properties) {
		this.propertiesFile = properties;
	}

	public URL getPropertiesFile() {
		return propertiesFile;
	}

	@Optional
	@RunTime
	@CreoleParameter(comment = "Output annotation set name", defaultValue = "")
	public void setOutputASName(String outputASName) {
		this.outputASName = outputASName;
	}

	public String getOutputASName() {
		return this.outputASName;
	}

}
