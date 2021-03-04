import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class LabelCombine {

	public static void main(String[] args) throws FileNotFoundException {
		// load the label file into memory
		System.out.println("Loading labels...");
		final List<String> labels = load("/media/veracrypt2/TUDelft/labels/msmarco-doctrain-qrels.tsv");
		System.out.println("Loading scores...");
		final List<String> scores = load("/media/veracrypt2/TUDelft/labels/msmarco-doctrain-top100");

		System.out.println("Mapping scores...");
		final Map<String, Integer> queryDocToScore = new HashMap<>();
		for (String line : labels) {
			final String[] split = line.split(" ");
			final String queryId = split[0];
			final String docId = split[2];
			final String uniqueName = queryId + "_" + docId;
			final int score = Integer.parseInt(split[3]);
			queryDocToScore.put(uniqueName, score);
		}

		System.out.println("Building output file...");
		try (PrintWriter w = new PrintWriter("/media/veracrypt2/TUDelft/labels/run1.txt")) {
			for (String s : scores) {
				final String[] split = s.split(" ");
				final String queryId = split[0];
				final String docId = split[2];
				final String number = split[3];
				final String score = split[4];
				final String name = split[5];

				final String uniqueName = queryId + "_" + docId;
				Integer label = queryDocToScore.get(uniqueName);
				if (label == null)
					label = 0;
				w.println(queryId + " " + label + " " + docId + " " + number + " " + score + " " + name);
			}
		}
		System.out.println("Done!");
	}

	private static List<String> load(String path) throws FileNotFoundException {
		final List<String> lines = new ArrayList<>();
		try (Scanner sc = new Scanner(new File(path))) {
			while (sc.hasNextLine()) {
				lines.add(sc.nextLine());
			}
		}
		return lines;
	}

}
