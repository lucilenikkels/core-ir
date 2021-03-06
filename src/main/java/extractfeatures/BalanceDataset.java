package extractfeatures;

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

	// DO NOT USE - this changes the order, which toolkits seem to make use of, only for testing!
	public static void main(String[] args) throws FileNotFoundException {
		System.out.println("Started...");
		final List<String> ones = new ArrayList<>();
		final List<String> zeros = new ArrayList<>();

		String input = args[0];
		String output = args[1];

		try (Scanner sc = new Scanner(new File(input))) {
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
		try (PrintWriter w = new PrintWriter(new File(output))) {
			for (String line : combined) {
				w.println(line);
			}
		}
	}
}
