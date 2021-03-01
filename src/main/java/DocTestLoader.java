import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class DocTestLoader {

	public static Map<String, String[]> load() throws FileNotFoundException {
		System.out.println("Loading doctest file...");
		final Map<String, String[]> queryToDocuments = new HashMap<>();
		final File f = new File("/home/coreir/lm_model/msmarco-doctest2019-top100");
		try (Scanner sc = new Scanner(f)) {
			while (sc.hasNextLine()) {
				final String line = sc.nextLine();
				// format: 1108939 Q0 D388799 1 -4.80563 IndriQueryLikelihood
				final String[] split = line.split(" ");
				final String queryId = split[0];
				final String documentId = split[2];
				final int position = Integer.parseInt(split[3]) - 1;

				String[] docs = queryToDocuments.get(queryId);
				if (docs == null)
					queryToDocuments.put(queryId, docs = new String[100]);
				docs[position] = documentId;
			}
		}
		return queryToDocuments;
	}
}
