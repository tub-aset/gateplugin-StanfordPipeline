package gate.stanfordnlp;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;

import org.apache.log4j.Logger;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import gate.AnnotationSet;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.ExecutionException;
import gate.creole.metadata.CreoleParameter;
import gate.creole.metadata.Optional;
import gate.creole.metadata.RunTime;

public abstract class StanfordAnnotatorAnalyser extends AbstractLanguageAnalyser {
	private static final long serialVersionUID = 3038786580835951910L;
	private static Logger logger = Logger.getLogger(StanfordAnnotatorAnalyser.class);

	protected Annotator pipeline;
	protected String outputASName;

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

			AnnotationMapper mapper = new AnnotationMapper(outputAnnotationSet);
			mapper.addGateAnnotations(annotation);
		} catch (Exception e) {
			throw new ExecutionException(e);
		}
	}

	protected Annotation annotateContent(String content) {
		Annotation annotation = new Annotation(content);
		pipeline.annotate(annotation);
		return annotation;
	}

	protected Properties loadProperties(URL propertiesUrl) throws IOException {
		Properties props = new Properties();
		if (propertiesUrl != null) {
			URLConnection connection = propertiesUrl.openConnection();
			InputStream inputStream = connection.getInputStream();
			props.load(inputStream);
			inputStream.close();
		}
		return props;
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
