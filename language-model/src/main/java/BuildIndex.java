import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import lemurproject.indri.IndexEnvironment;

public class BuildIndex {

	public static void main(String[] args) {
		try {
			if (false) {
				System.out.println("Prevent running by accident and overwrite index");
				System.exit(0);
			}
			System.out.println("Building index... This could take a while...");
			IndexEnvironment env = new IndexEnvironment();
			env.setStemmer("porter");
			env.setStoreDocs(true);
			env.setIndexedFields(new String[] {"mainbody", "heading", "inlink", "body", "title", "table", "td", "a",
					"applet", "object", "embed"});
			env.setStopwords(getStopWords());

			final String[] fields = new String[] { "docno", "url" };
			env.setMetadataIndexedFields(fields, fields);

			env.create("E:/core-ir/msmarcofull.idx");
			env.addFile("E:/core-ir/msmarco-docs.trec", "trectext");
			env.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static String[] getStopWords() throws FileNotFoundException {
		final List<String> words = new ArrayList<>();
		try (Scanner sc = new Scanner(new File("./nltk_stopwords.txt"))) {
			while (sc.hasNextLine()) {
				final String line = sc.nextLine();
				if (line == null || line.equals(""))
					continue;
				words.add(line);
			}
		}
		System.out.println("Loaded " + words.size() + " stop word(s)... " + words);
		return words.toArray(new String[0]);
	}
}
