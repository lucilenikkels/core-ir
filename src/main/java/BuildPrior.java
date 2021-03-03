import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import lemurproject.indri.ParsedDocument;
import lemurproject.indri.QueryEnvironment;

public class BuildPrior {

	private static final int CHUNK_SIZE = 2048;

	// calculate prior P(D), the probability of the document D
	public static void main(String[] strings) throws Exception {
		// go through the documents and match their id with a prior probability
		final QueryEnvironment env = new QueryEnvironment();
		// by default, Indri uses a query likelihood function with Dirichlet prior
		// smoothing to weight terms
		env.setStopwords(BuildIndex.getStopWords());
		System.out.println("Loading indexes...");
		env.addIndex("/home/coreir/lm_model/msmarco.idx");
		System.out.println("Environment loaded, found " + env.documentCount() + " document(s).");

		final List<Integer> all = new ArrayList<>();
		for (int i = 1; i <= env.documentCount(); i++)
			all.add(i);
		final List<List<Integer>> batches = partition(all, CHUNK_SIZE);

		System.out.println("Finding document lengths...");
		final Map<String, Double> docPriors = new HashMap<>();
		for (int batchId = 0; batchId < batches.size(); batchId++) {
			final List<Integer> batch = batches.get(batchId);
			System.out.println("Processing batch " + (batchId + 1) + "/" + batches.size() + "...");
			final ParsedDocument[] pds = env.documents(toPrimitive(batch));
			for (int i = 0; i < pds.length; i++) {
				final ParsedDocument doc = pds[i];
				// final String id = RunQuery.parseDocumentId(doc);

				// System.out.println("Lookup ID " + id + " meta " + doc.metadata);
				// final int res = env.documentIDsFromMetadata("docno", new String[] { id })[0];
				// if (res != batch.get(i))
				// 	throw new RuntimeException(
				//			"Batch index " + i + " matched to docno " + id + " which matched back to " + res);
				final double prior = calculatePrior(doc);
				docPriors.put(Integer.toString(batch.get(i)), prior);
			}
		}

		Double highestPrior = null;
		for (double d : docPriors.values())
			if (highestPrior == null || d > highestPrior)
				highestPrior = d;
		// System.out.println("total " + totalPrior);

		System.out.println("Building prior map...");
		try (PrintWriter writerPrior = new PrintWriter(new File("./prior_values.dat"))) {
			for (Entry<String, Double> en : docPriors.entrySet()) {
				final String doc = en.getKey();
				double prior = en.getValue();
				prior /= highestPrior; // scale it
				writerPrior.println(doc + " " + Math.log(prior));
			}
		}

		for (ParsedDocument s : env.documents(new int[] { 1, 2, 3, 4, 5, 6 })) {
			System.out.println(s.metadata);
		}
	}

	private static double calculatePrior(ParsedDocument doc) {
		// simple prior, document length
		return doc.content.length();
	}

	public static int[] toPrimitive(List<Integer> batch) {
		final int[] res = new int[batch.size()];
		for (int i = 0; i < res.length; i++)
			res[i] = batch.get(i);
		return res;
	}

	public static List<List<Integer>> partition(Collection<Integer> members, int maxSize) {
		final List<List<Integer>> res = new ArrayList<>();
		List<Integer> internal = new ArrayList<>();

		for (Integer member : members) {
			internal.add(member);

			if (internal.size() == maxSize) {
				res.add(internal);
				internal = new ArrayList<>();
			}
		}
		if (!internal.isEmpty())
			res.add(internal);
		return res;
	}
}
