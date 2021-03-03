import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import lemurproject.indri.ParsedDocument;
import lemurproject.indri.QueryEnvironment;
import lemurproject.indri.ScoredExtentResult;

public class RunQuery {

	private static final int LIMIT = 100;
	private static final String[] PRIORS = new String[] { //
			"", //
			"#prior(doclength) ", //
	};

	public static void main(String[] args) throws Exception {
		final QueryEnvironment env = new QueryEnvironment();
		// by default, Indri uses a query likelihood function with Dirichlet prior
		// smoothing to weight terms
		env.setStopwords(BuildIndex.getStopWords());
		System.out.println("Loading indexes...");
		env.addIndex("/home/coreir/lm_model/msmarco.idx");
		System.out.println("Environment loaded, found " + env.documentCount() + " document(s).");

		final Map<String, String[]> theirSelections = DocTestLoader.load();

		// build output files
		final File resultDir = new File("./results/");
		resultDir.mkdirs();
		final PrintWriter[] writers = new PrintWriter[PRIORS.length];
		for (int i = 0; i < PRIORS.length; i++) {
			final String prior = PRIORS[i];
			final String outFile = "result_" + prior.replaceAll("[^a-zA-Z0-9\\s]", "") + "_" + new Date().getTime()
					+ ".out";
			writers[i] = new PrintWriter(new File(resultDir, outFile));
		}

		// now we traverse all queries individually from msmarco-test2019-queries.tsv
		final List<Query> queries = loadQueries();
		for (Query query : queries) {
			System.out.println();
			System.out.println("Processing query [" + query.getId() + "] \"" + query.getWords() + "\"");

			final String[] theirSelection = theirSelections.get(query.getId());
			if (theirSelection.length != LIMIT)
				throw new IllegalArgumentException("Their selection did not have " + LIMIT + " sample(s)");

			// System.out.println("THEIR " + Arrays.toString(theirSelection));

			for (int priorId = 0; priorId < PRIORS.length; priorId++) {
				final String prior = PRIORS[priorId];

				final String[] ourSelection = new String[LIMIT];
				final ScoredExtentResult[] res = env.runQuery(//
						"#combine(" + prior + query.getWords() + ")", LIMIT); // 
				// System.out.println("Query count was " + res.length + " looking up documents...");

				final ParsedDocument[] docs = env.documents(res);

				for (int i = 0; i < res.length; i++) {
					final ScoredExtentResult score = res[i];
					final double logProbability = score.score;
					/*
					 * Indri returns the log of the actual probability value. log(0) equals negative
					 * infinity, and log(1) equals zero, so Indri document scores are always
					 * negative.
					 */
					// final double actualProbability = Math.pow(Math.E, score.score);
					final ParsedDocument doc = docs[i];
					final String documentId = parseDocumentId(doc); // + "_" + logProbability;
					ourSelection[i] = documentId;

					final String line = query.getId() + " Q0 " + documentId + " " + (i + 1) + " " + logProbability
							+ " IndriQueryLikelihood";
					final PrintWriter w = writers[priorId];
					w.println(line);
					w.flush();
				}

				// compare the two
				// System.out.println("OUR [" + prior + "] " + Arrays.toString(ourSelection));
				final double acc = compare(Arrays.asList(ourSelection), Arrays.asList(theirSelection));
				System.out.println("PRIOR [" + prior + "] ACCURACY [" + acc + "]");
			}
		}
	}

	// NDCG@10 we should get around 0.5
	// TODO implement better check
	private static double compare(List<String> our, List<String> their) {

		int correct = 0;
		for (String s : their) {
			// System.out.println("Check " + s + " was in " + our.subList(0, 8));
			// System.exit(0);
			if (our.contains(s)) {
				correct++;
			}
		}
		final double acc = (correct * 100.0 / their.size());
		// System.out.println("Accuracy match " + acc + "%");
		return acc;
	}

	public static String parseDocumentId(ParsedDocument d) {
		final String id = new String((byte[]) d.metadata.get("docno")); // new String((byte[]) , StandardCharsets.UTF_8);
		// System.out.println("id [" + id + "]");

		return id.substring(0, id.length() - 1);
	}

	private static List<Query> loadQueries() throws FileNotFoundException {
		final List<Query> lines = new ArrayList<>();
		final File f = new File("/home/coreir/lm_model/msmarco-test2019-queries.tsv");
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
