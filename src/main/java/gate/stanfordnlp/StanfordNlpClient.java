package gate.stanfordnlp;

import static gate.stanfordnlp.Util.hasValue;
import static gate.stanfordnlp.Util.noValue;

import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;

import org.apache.log4j.Logger;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLPClient;
import gate.AnnotationSet;
import gate.Resource;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleParameter;
import gate.creole.metadata.CreoleResource;
import gate.creole.metadata.Optional;
import gate.creole.metadata.RunTime;

@CreoleResource(name = "StanfordNlpClient", comment = "This is a interface for StanfordCoreNLPClient")
public class StanfordNlpClient extends AbstractLanguageAnalyser {
	private static final long serialVersionUID = -6194526132377800058L;
	private static Logger logger = Logger.getLogger(StanfordNlpClient.class);

	private String host;
	private Integer port;
	private Integer threads;
	private String apiKey;
	private String apiSecret;

	private URL propertiesFile;
	private String outputASName;

	private StanfordCoreNLPClient pipeline;

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
		try {
			if (noValue(host) && port == null && threads == null && noValue(apiKey) && noValue(apiSecret)) {
				pipeline = new StanfordCoreNLPClient(props);
			} else if (hasValue(host) && port != null && threads == null && noValue(apiKey) && noValue(apiSecret)) {
				pipeline = new StanfordCoreNLPClient(props, host, port);
			} else if (hasValue(host) && port != null && threads != null && noValue(apiKey) && noValue(apiSecret)) {
				pipeline = new StanfordCoreNLPClient(props, host, port, threads);
			} else if (hasValue(host) && port == null && threads == null && hasValue(apiKey) && hasValue(apiSecret)) {
				pipeline = new StanfordCoreNLPClient(props, host, apiKey, apiSecret);
			} else if (hasValue(host) && port != null && threads == null && hasValue(apiKey) && hasValue(apiSecret)) {
				pipeline = new StanfordCoreNLPClient(props, host, port, apiKey, apiSecret);
			} else if (hasValue(host) && port != null && threads != null && hasValue(apiKey) && hasValue(apiSecret)) {
				pipeline = new StanfordCoreNLPClient(props, host, port, threads, apiKey, apiSecret);
			} else {
				throw new IllegalStateException(
						"invalid host, port, threads, apiKey or apiSecret values. see edu.stanford.nlp.pipeline.StanfordCoreNLPClient.StanfordCoreNLPClient");
			}
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
	@CreoleParameter(comment = "StanfordNLPClient host property (if not specified, StanfordNLPClient will use environment variable 'CORENLP_HOST')")
	public void setHost(String host) {
		this.host = host;
	}

	public String getHost() {
		return host;
	}

	@Optional
	@CreoleParameter(comment = "StanfordNLPClient host property (if not specified, StanfordNLPClient will use '80' or '443')")
	public void setPort(Integer port) {
		this.port = port;
	}

	public Integer getPort() {
		return port;
	}

	@Optional
	@CreoleParameter(comment = "StanfordNLPClient port property (if not specified, StanfordNLPClient will use 1 thread)")
	public void setThreads(Integer threads) {
		this.threads = threads;
	}

	public Integer getThreads() {
		return threads;
	}

	@Optional
	@CreoleParameter(comment = "StanfordNLPClient host property (if not specified, StanfordNLPClient will use environment variable 'CORENLP_KEY')")
	public void setApiKey(String apiKey) {
		this.apiKey = apiKey;
	}

	public String getApiKey() {
		return apiKey;
	}

	@Optional
	@CreoleParameter(comment = "StanfordNLPClient host property (if not specified, StanfordNLPClient will use environment variable 'CORENLP_SECRET')")
	public void setApiSecret(String apiSecret) {
		this.apiSecret = apiSecret;
	}

	public String getApiSecret() {
		return apiSecret;
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
