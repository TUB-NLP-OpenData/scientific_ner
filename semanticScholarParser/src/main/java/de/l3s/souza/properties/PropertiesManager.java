package de.l3s.souza.properties;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class PropertiesManager {

	private int maxAbstracts;
	private String gzipFolder;
	private String outputFileName;
	private String inputVenues;

	public String getOutputFileName() {
		return outputFileName;
	}

	public void setOutputFileName(String outputFileName) {
		this.outputFileName = outputFileName;
	}

	private int minimumYear;

	public int getMaxAbstracts() {
		return maxAbstracts;
	}

	public void setMaxAbstracts(int maxAbstracts) {
		this.maxAbstracts = maxAbstracts;
	}

	public String getGzipFolder() {
		return gzipFolder;
	}

	public void setGzipFolder(String gzipFolder) {
		this.gzipFolder = gzipFolder;
	}

	public String getInputVenues() {
		return inputVenues;
	}

	public void setInputVenues(String inputVenues) {
		this.inputVenues = inputVenues;
	}

	public int getMinimumYear() {
		return minimumYear;
	}

	public void setMinimumYear(int minimumYear) {
		this.minimumYear = minimumYear;
	}

	public PropertiesManager () {

		readPropFile();

	}

	private void readPropFile ()
	{
		Properties prop = new Properties();
		InputStream input = null;

		try {

			String filename = "config.properties";
			input = PropertiesManager.class.getClassLoader().getResourceAsStream(filename);
			if(input==null){
				System.out.println("Sorry, unable to find " + filename);
				return;
			}
			
			// load a properties file
			prop.load(input);

			gzipFolder = prop.getProperty("gzip.input.folder");
			minimumYear = Integer.parseInt(prop.getProperty("minimum.year"));
			maxAbstracts = Integer.parseInt(prop.getProperty("max.abstracts.sample"));
			inputVenues = prop.getProperty("text.input.venues");
			outputFileName = prop.getProperty("output.filename");

		} catch (IOException ex) {
			ex.printStackTrace();
		} finally {
			if (input != null) {
				try {
					input.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}	
}
