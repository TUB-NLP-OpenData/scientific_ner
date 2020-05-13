package de.l3s.souza.semanticscholarindex;

import java.io.IOException;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class AppTest 
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public AppTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( AppTest.class );
    }

    /**
     * Rigourous Test :-)
     * @throws IOException 
     */
    public void testApp() throws IOException
    {
//    	SemanticScholarJsonStream test = new SemanticScholarJsonStream ("/Users/tarcisio/Documents/Promotion/SciERC/dataset/venues.txt");
    	
//    	test.fileReader("/Users/tarcisio/Documents/Promotion/SciERC/dataset/samplenewvenues/","papers.txt");
    	
    	SemanticScholarJsonStream test = new SemanticScholarJsonStream ();
    	
    	//test.samplePapers("/Users/tarcisio/Documents/Promotion/SciERC/dataset/samplenewvenues/", "sample.txt");
    	
    	//System.out.println("total papers: " + test.getTotalPapers());
    }
}
