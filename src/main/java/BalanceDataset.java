import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class BalanceDataset {

	private static final Random RANDOM = new Random();

	public static void main(String[] args) throws FileNotFoundException {
		System.out.println("Started...");
		final List<String> ones = new ArrayList<>();
		final List<String> zeros = new ArrayList<>();

		try (Scanner sc = new Scanner(new File("/media/veracrypt2/TUDelft/features.txt"))) {
			while (sc.hasNextLine()) {
				final String line = sc.nextLine();
				if (line.startsWith("1 ")) {
					ones.add(line);
				} else if (line.startsWith("0 ")) {
					zeros.add(line);
				} else {
					throw new RuntimeException();
				}
			}
		}

		System.out.println("Ones " + ones.size());
		System.out.println("Zeros " + zeros.size());
		final List<String> newZeros = new ArrayList<>();
		for (int i = 0; i < ones.size(); i++) {
			newZeros.add(zeros.get(RANDOM.nextInt(zeros.size())));
		}

		System.out.println("Building...");
		final List<String> combined = new ArrayList<>();
		combined.addAll(ones);
		combined.addAll(newZeros);
		Collections.shuffle(combined);
		try (PrintWriter w = new PrintWriter(new File("/media/veracrypt2/TUDelft/features_balanced.txt"))) {
			for (String line : combined) {
				w.println(line);
			}
		}
	}
}
