import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import lemurproject.indri.ParsedDocument;
import lemurproject.indri.QueryEnvironment;
import lemurproject.indri.ScoredExtentResult;

public class RunQuery {

	private static final int LIMIT = 100;

	public static void main(String[] args) throws Exception {
		final QueryEnvironment env = new QueryEnvironment();
		// by default, Indri uses a query likelihood function with Dirichlet prior
		// smoothing to weight terms
		env.setStopwords(BuildIndex.getStopWords());
		System.out.println("Loading indexes...");
		env.addIndex("/media/veracrypt2/TUDelft/msmarco.idx");
		System.out.println("Environment loaded, found " + env.documentCount() + " document(s).");

		final Map<String, String[]> theirSelections = DocTestLoader.load();

		// now we traverse all queries individually from msmarco-test2019-queries.tsv
		final List<Query> queries = loadQueries();
		for (Query query : queries) {
			System.out.println();
			System.out.println("Processing query [" + query.getId() + "] \"" + query.getWords() + "\"");

			final String[] theirSelection = theirSelections.get(query.getId());
			if (theirSelection.length != LIMIT)
				throw new IllegalArgumentException("Their selection did not have " + LIMIT + " sample(s)");

			final String[] ourSelection = new String[LIMIT];

			final ScoredExtentResult[] res = env.runQuery(query.getWords(), LIMIT);
			System.out.println("Query count was " + res.length + " looking up documents...");

			final ParsedDocument[] docs = env.documents(res);

			for (int i = 0; i < res.length; i++) {
				// final ScoredExtentResult score = res[i];
				// final double logProbability = score.score;
				/*
				 * Indri returns the log of the actual probability value. log(0) equals negative
				 * infinity, and log(1) equals zero, so Indri document scores are always
				 * negative.
				 */
				// final double actualProbability = Math.pow(Math.E, score.score);
				final ParsedDocument doc = docs[i];
				final String documentId = parseDocumentId(doc);
				ourSelection[i] = documentId;
			}

			// compare the two
			compare(Arrays.asList(ourSelection), Arrays.asList(theirSelection));
		}
	}

	// NDCG@10 we should get around 0.5
	// TODO implement better check
	private static void compare(List<String> our, List<String> their) {
		int correct = 0;
		for (String s : their) {
			// System.out.println("Check " + s + " was in " + our.subList(0, 8));
			// System.exit(0);
			if (our.contains(s)) {
				correct++;
			}
		}
		System.out.println("Accuracy match " + (correct * 100 / their.size()) + "%");
	}

	private static String parseDocumentId(ParsedDocument d) {
		String value = new String((byte[]) d.metadata.get("docno"));
		value = value.substring(0, value.length() - 1);
		// System.out.println("Len " + value.length());
		// System.out.println("Document ID \"" + value + ".");
		return value;
	}

	private static List<Query> loadQueries() throws FileNotFoundException {
		final List<Query> lines = new ArrayList<>();
		final File f = new File("/media/veracrypt2/TUDelft/msmarco-test2019-queries.tsv");
		try (Scanner sc = new Scanner(f)) {
			while (sc.hasNextLine()) {
				final String line = sc.nextLine();
				final String[] s = line.split("\t");
				final String words = s[1].replaceAll("[^a-zA-Z0-9\\s]", "");
				lines.add(new Query(s[0], words));
			}
		}
		return lines;
	}
}
