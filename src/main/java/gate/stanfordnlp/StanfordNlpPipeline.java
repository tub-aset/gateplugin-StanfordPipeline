package gate.stanfordnlp;

import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;

import org.apache.log4j.Logger;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.StringUtils;
import gate.AnnotationSet;
import gate.Factory;
import gate.Factory.DuplicationContext;
import gate.Gate;
import gate.Resource;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.AbstractResource;
import gate.creole.CustomDuplication;
import gate.creole.ExecutionException;
import gate.creole.ResourceData;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleParameter;
import gate.creole.metadata.CreoleResource;
import gate.creole.metadata.Optional;
import gate.creole.metadata.RunTime;

@CreoleResource(name = "StanfordNlpPipeline", comment = "This is a simple Stanford NLP Pipeline")
public class StanfordNlpPipeline extends AbstractLanguageAnalyser implements CustomDuplication {
	private static final long serialVersionUID = -474306293452776717L;
	private static Logger logger = Logger.getLogger(StanfordNlpPipeline.class);

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
	public Resource duplicate(DuplicationContext ctx) throws ResourceInstantiationException {
		ResourceData resourceData = Gate.getCreoleRegister().get(StanfordNlpPipeline.class.getCanonicalName());
		StanfordNlpPipeline duplicate = new StanfordNlpPipeline();

		duplicate.setName(resourceData.getName() + "_" + Gate.genSym());
		AbstractResource.setParameterValues(duplicate, getInitParameterValues());
		AbstractResource.setParameterValues(duplicate, getRuntimeParameterValues());
		duplicate.setFeatures(Factory.newFeatureMap());
		duplicate.getFeatures().putAll(getFeatures());

		duplicate.pipeline = pipeline;

		resourceData.addInstantiation(duplicate);
		return duplicate;
	}

	@Override
	public void execute() throws ExecutionException {
		try {
			AnnotationSet outputAnnotationSet = document.getAnnotations(outputASName);

			Annotation annotation = annotateContent(document.getContent().toString());

			AnnotationMapper.addGateAnnotations(outputAnnotationSet, annotation);
		} catch (Exception e) {
			throw new ExecutionException(e);
		}
	}

	private Annotation annotateContent(String content) {
		Annotation annotation = new Annotation(content);
		pipeline.annotate(annotation);
		return annotation;
	}

	@Optional
	@CreoleParameter(comment = "StanfordNLP pipeline annotators (overrides annotators property from properties and propertiesFile)", defaultValue = "tokenize,ssplit,pos,lemma,ner,parse,coref")
	public void setAnnotators(String annotators) {
		this.annotators = annotators;
	}

	public String getAnnotators() {
		return annotators;
	}

	@Optional
	@CreoleParameter(comment = "StanfordNLP pipeline properties (command-line style, e.g. -ner.useSUTime 0)", defaultValue = "")
	public void setProperties(String properties) {
		this.properties = properties;
	}

	public String getProperties() {
		return properties;
	}

	@Optional
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
