package gate.stanfordnlp;

import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefChain.CorefMention;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.OriginalTextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.ValueAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
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
import gate.util.InvalidOffsetException;

@CreoleResource(name = "StanfordPipeline", comment = "This is a simple Stanford Pipeline")
public class StanfordPipeline extends AbstractLanguageAnalyser {

	private static final String ANNOTATION_MENTION_NAME = "mention";
	private static final String ANNOTATION_MENTION_FEATURE_ANIMACY_NAME = "animacy";
	private static final String ANNOTATION_MENTION_FEATURE_MENTIONTYPE_NAME = "type";
	private static final String ANNOTATION_MENTION_FEATURE_REPRESENTATIVE_NAME = "representative";
	private static final String ANNOTATION_MENTION_FEATURE_ISREPRESENTATIVE_NAME = "isrepresentative";

	private static final String RELATION_COREF_NAME = "coref";

	private static final Class<?>[] EXCLUDE_DEFAULT_ANNOTATIONS = new Class[] { TextAnnotation.class,
			OriginalTextAnnotation.class, ValueAnnotation.class, CorefChain.class, CorefChainAnnotation.class };

	private static final long serialVersionUID = 1L;

	private String annotators;
	private URL properties;
	private String outputASName;

	private StanfordCoreNLP pipeline;

	@Override
	public Resource init() throws ResourceInstantiationException {
		Properties props = new Properties();
		props.setProperty("ner.useSUTime", "0");
		if (properties != null) {
			try {
				URLConnection connection = properties.openConnection();
				InputStream inputStream = connection.getInputStream();
				props.load(inputStream);
				inputStream.close();
			} catch (Exception e) {
				throw new ResourceInstantiationException(e);
			}
		}
		if (annotators.length() > 0) {
			props.setProperty("annotators", annotators);
		}
		pipeline = new StanfordCoreNLP(props);
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
			addCoreAnnotation(outputAnnotationSet, annotation, annotation.getClass(), EXCLUDE_DEFAULT_ANNOTATIONS);
			addCorefAnnotations(outputAnnotationSet, annotation);
		} catch (Exception e) {
			throw new ExecutionException(e);
		}
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	protected void addCoreAnnotation(AnnotationSet outputAnnotationSet, CoreMap annotation, Class<?> keyAnnotationClass,
			Class<?>... excludeKeyClasses) throws InvalidOffsetException {
		FeatureMap map = Factory.newFeatureMap();
		for (Class<?> clazz : annotation.keySet()) {
			Object object = annotation.get((Class) clazz);
			if (object instanceof Iterable<?>) {
				Iterable<?> iterable = (Iterable<?>) object;
				List<Object> list = new ArrayList<>();
				for (Object item : iterable) {
					if (item instanceof CoreMap) {
						addCoreAnnotation(outputAnnotationSet, (CoreMap) item, clazz, excludeKeyClasses);
					} else {
						list.add(item);
					}
				}
				if (!list.isEmpty()) {
					addToFeatures(map, clazz, list);
				}

			} else if (!isAssignableFromAny(clazz, excludeKeyClasses)) {
				addToFeatures(map, clazz, object);
			}
		}
		if (annotation.keySet().contains(CharacterOffsetBeginAnnotation.class)
				&& annotation.keySet().contains(CharacterOffsetEndAnnotation.class)) {
			Long start = annotation.get(CharacterOffsetBeginAnnotation.class).longValue();
			Long end = annotation.get(CharacterOffsetEndAnnotation.class).longValue();
			String type = getGateName(keyAnnotationClass);
			AnnotationSet others = outputAnnotationSet.getCovering(type, start, end);
			if (others.size() == 1) {
				others.iterator().next().getFeatures().putAll(map);
			} else {
				outputAnnotationSet.add(start, end, type, map);
			}
		}
	}

	private String getGateName(Class<?> clazz) {
		String key = clazz.getSimpleName();
		if (key.endsWith("Annotation")) {
			key = key.substring(0, key.length() - "Annotation".length());
		}
		return key;
	}

	private void addToFeatures(FeatureMap map, Class<?> clazz, Object object) {
		map.put(getGateName(clazz), object);
	}

	protected void addCorefAnnotations(AnnotationSet outputAnnotationSet, Annotation annotation)
			throws InvalidOffsetException {
		if (annotation.keySet().contains(CorefCoreAnnotations.CorefChainAnnotation.class)) {
			List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
			for (CorefChain cc : annotation.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
				List<Integer> relationIds = new ArrayList<>();
				for (CorefMention mention : cc.getMentionsInTextualOrder()) {
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
							ANNOTATION_MENTION_NAME, map);
				}
				if (!relationIds.isEmpty()) {
					outputAnnotationSet.getRelations().addRelation(RELATION_COREF_NAME, toIntArray(relationIds));
				}
			}
		}
	}

	private Annotation annotateContent(String content) {
		Annotation annotation = new Annotation(content);
		pipeline.annotate(annotation);
		return annotation;
	}

	private static int[] toIntArray(List<Integer> list) {
		int[] ret = new int[list.size()];
		int i = 0;
		for (Integer e : list)
			ret[i++] = e.intValue();
		return ret;
	}

	private static boolean isAssignableFromAny(Class<?> clazz, Class<?>... others) {
		for (Class<?> class1 : others) {
			if (clazz.isAssignableFrom(class1))
				return true;
		}
		return false;
	}

	@CreoleParameter(comment = "Stanford annotators", defaultValue = "tokenize,ssplit,pos,lemma,ner,parse,mention,coref")
	public void setAnnotators(String annotators) {
		this.annotators = annotators;
	}

	public String getAnnotators() {
		return annotators;
	}

	@CreoleParameter(comment = "properties file for stanford pipeline")
	public void setProperties(URL properties) {
		this.properties = properties;
	}

	public URL getProperties() {
		return properties;
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
