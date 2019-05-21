package gate.stanfordnlp;

import java.io.IOException;
import java.net.URL;
import java.util.Properties;

import org.apache.log4j.Logger;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.StringUtils;
import gate.Factory;
import gate.Factory.DuplicationContext;
import gate.Gate;
import gate.Resource;
import gate.creole.AbstractResource;
import gate.creole.CustomDuplication;
import gate.creole.ResourceData;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleParameter;
import gate.creole.metadata.CreoleResource;
import gate.creole.metadata.Optional;

@CreoleResource(name = "StanfordNlpPipeline", comment = "This is a simple Stanford NLP Pipeline")
public class StanfordNlpPipeline extends StanfordAnnotatorAnalyser implements CustomDuplication {
	private static final long serialVersionUID = -474306293452776717L;
	private static Logger logger = Logger.getLogger(StanfordNlpPipeline.class);

	private String annotators;
	private String properties;
	private URL propertiesFile;

	@Override
	public Resource init() throws ResourceInstantiationException {
		Properties props;
		try {
			props = loadProperties(propertiesFile);
		} catch (IOException e) {
			throw new ResourceInstantiationException(e);
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

}
