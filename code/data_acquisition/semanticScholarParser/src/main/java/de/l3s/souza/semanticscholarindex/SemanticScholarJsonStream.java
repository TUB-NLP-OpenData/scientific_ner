package de.l3s.souza.semanticscholarindex;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import org.json.JSONObject;

import de.l3s.souza.properties.PropertiesManager;

public class SemanticScholarJsonStream 
{
	public SemanticScholarJsonStream(String venuesPathFile) {
		this.venuesPathFile = venuesPathFile;
	}
	
	public SemanticScholarJsonStream() {
		
	}

	private String venuesPathFile;
	private static int maxAbstracts;
	private static PropertiesManager pm;
	private static Writer writer;
	private int totalAbstracts = 0;
	private HashMap<String,Integer> papersPerVenue = new HashMap<String,Integer> ();
	private ArrayList<JSONObject> papers = new ArrayList <JSONObject>();
			
	private void samplePapers (String outputFile) throws IOException
	{
		
	//	File inputFolder = new File (path);
		ArrayList<String> venues = new ArrayList<String>();
		
		writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(outputFile, false), "utf-8"),
					32 * 1024);

		totalAbstracts = 0;
	    Random random = new Random (System.currentTimeMillis());
	    int totalgenerated = 0;	    
	    while (totalgenerated < maxAbstracts && (!papers.isEmpty()))
	    {
	    	int randomNumberGenerated = random.nextInt(papers.size());

	    	writer.write(papers.get(randomNumberGenerated).toString() + "\n");
	    	totalAbstracts++;
	    	
	    	String key = papers.get(randomNumberGenerated).getString("venue").toString();

	    	if (papersPerVenue.containsKey(key))
	    	{
	    		int value = papersPerVenue.get(key);
	    		value = value + 1;
	    		papersPerVenue.put(key, value);
	    	}
	    	else
	    		papersPerVenue.put(key, 1);

	    	papers.remove(randomNumberGenerated);

	    	totalgenerated ++;

	    }

	    System.out.println("Stats for " + totalAbstracts + " abstracts randomly generated");
	    for (Entry<String,Integer> v : papersPerVenue.entrySet())
	    	System.out.println("Venue: " + v.getKey() + " #Papers: " + v.getValue());
	    
	    for (int i=0;i<venues.size();i++)
    		System.out.println(venues.get(i));
		
	    try {
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	}
	
    public void findAbstracts( String path, String outputFile ) throws IOException
    {
        File inputFolder = new File (path);
        int totalFilesRead = 0;
        boolean matchFound = false;
        
       /* writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(outputFile.replace(".txt", fileNumber + ".txt"), false), "utf-8"),
				32 * 1024);
        */
        File inputVenues = new File (venuesPathFile);
        ArrayList<String> venues = new ArrayList<String>();
        FileReader fr = new FileReader (inputVenues);
        String line = "";
        BufferedReader brVenues = new BufferedReader (fr);
        
        while ((line = brVenues.readLine()) != null) 
        	venues.add(line);
        
        System.out.println("Reading dataset...");
        
        for (File f:inputFolder.listFiles())
        {
        	if (!f.getName().contains(".gz"))
				continue;
        	matchFound = false;
        	InputStream fileStream = new FileInputStream (f.getAbsolutePath());
        	InputStream gzipstream = new GZIPInputStream (fileStream);
        	Reader decoder = new InputStreamReader(gzipstream, "UTF-8");
        	
        	BufferedReader br = new BufferedReader (decoder);
        	line = "";
        	totalFilesRead++;
        /*	if (totalFilesRead % 10 == 0)
        		System.out.println("Total files processed " + totalFilesRead);*/
        	int n = 0;
        	
        	while ((line = br.readLine()) != null) {
        		
        		n++;
   	         	JSONObject json = new JSONObject(line);
        		
   	         	for (int i=0;i<venues.size();i++)
   	         	{
   	         		if (json.get("venue").toString().contains(venues.get(i)) && (json.getInt("year")>=pm.getMinimumYear()))
   	         		{
   	         			totalAbstracts++;
   	         			papers.add(json);
   	         			 if (!matchFound)
   	         			 {
   	         				System.out.println("Match found in file " + f.getName());
   	         				matchFound = true;
   	         			 }
   	         		}
   	         	}
        	}
        }
        
        System.out.println("Total of " + totalAbstracts + " abstracts found in repository that match venues list.");
        samplePapers (outputFile);
    }
    
    public static void main (String[]args) throws IOException
    {
    	
    	pm = new PropertiesManager ();
    	maxAbstracts = pm.getMaxAbstracts();
    	/*
    	System.out.println("USAGE:  venuesFilePath semanticScholarGzipFilesPath outputFileName");
    	SemanticScholarJsonStream ss = new SemanticScholarJsonStream (args[0]);
    	*/
    	
    	SemanticScholarJsonStream ss = new SemanticScholarJsonStream (pm.getInputVenues());
    	ss.findAbstracts(pm.getGzipFolder(),pm.getOutputFileName());
    	
    	System.out.println("Total abstracts: " + ss.getTotalAbstracts());
    }

	public int getTotalAbstracts() {
		return totalAbstracts;
	}

}
