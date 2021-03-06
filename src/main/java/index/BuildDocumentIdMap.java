package index;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import languagemodel.BuildPrior;
import languagemodel.RunQuery;
import lemurproject.indri.ParsedDocument;
import lemurproject.indri.QueryEnvironment;

public class BuildDocumentIdMap {

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
		final List<List<Integer>> batches = BuildPrior.partition(all, CHUNK_SIZE);

		System.out.println("Finding document ids...");
		try (PrintWriter writer = new PrintWriter(new File("./document_ids.dat"))) {
			for (int batchId = 0; batchId < batches.size(); batchId++) {
				final List<Integer> batch = batches.get(batchId);
				System.out.println("Processing batch " + (batchId + 1) + "/" + batches.size() + "...");
				final ParsedDocument[] pds = env.documents(BuildPrior.toPrimitive(batch));
				for (int i = 0; i < pds.length; i++) {
					final ParsedDocument doc = pds[i];
					final String id = RunQuery.parseDocumentId(doc);

					// System.out.println("Lookup ID " + id + " meta " + doc.metadata);
					// final int res = env.documentIDsFromMetadata("docno", new String[] { id })[0];
					// if (res != batch.get(i))
					// 	throw new RuntimeException(
					//			"Batch index " + i + " matched to docno " + id + " which matched back to " + res);
					writer.println(batch.get(i) + " " + id);
				}
			}
		}
		System.out.println("Done!");

	}

	public static Map<Integer, String> load() throws FileNotFoundException {
		final Map<Integer, String> res = new HashMap<>();
		try (Scanner sc = new Scanner(new File("./document_ids.dat"))) {
			while (sc.hasNextLine()) {
				final String line = sc.nextLine();
				final String[] split = line.split(" ");
				final int batchId = Integer.parseInt(split[0]);
				final String docId = split[1];
				res.put(batchId, docId);
			}
		}
		return res;
	}
}
