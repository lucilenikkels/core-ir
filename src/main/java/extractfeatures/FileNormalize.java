package extractfeatures;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class FileNormalize {

	public static void main(String[] args) throws FileNotFoundException {
		System.out.println("Reading...");
		final List<Sample> samples = new ArrayList<>();
		try (Scanner sc = new Scanner(new File("/media/veracrypt2/TUDelft/features.txt"))) {
			while (sc.hasNextLine()) {
				final String line = sc.nextLine();
				final String[] split = line.split(" ");
				final int rel = Integer.parseInt(split[0]);
				final String qid = split[1];
				final double[] features = new double[split.length - 2];
				for (int i = 2; i < split.length; i++) {
					features[i - 2] = Double.parseDouble(split[i].split(":")[1]);
				}
				samples.add(new Sample(rel, qid, features));
			}
		}
		System.out.println("Samples " + samples.size());

		final int features = samples.get(0).features.length;
		for (int featureIndex = 0; featureIndex < features; featureIndex++) {
			final List<Double> all = new ArrayList<>();
			for (Sample s : samples)
				all.add(s.features[featureIndex]);

			final double mean = mean(all);
			final double std = std(all);
			for (Sample s : samples) {
				final double old = s.features[featureIndex];
				s.features[featureIndex] = (old - mean) / std;
			}
		}

		System.out.println("Writing back to file...");

		try (PrintWriter w = new PrintWriter(new File("/media/veracrypt2/TUDelft/features_normalized.txt"))) {
			for (Sample line : samples) {
				w.println(line.asLine());
			}
		}
	}

	private static double std(List<Double> all) {
		double sum = 0.0, standardDeviation = 0.0;
		for (double num : all)
			sum += num;
		final double mean = sum / all.size();
		for (double num : all)
			standardDeviation += Math.pow(num - mean, 2);
		return Math.sqrt(standardDeviation / all.size());
	}

	private static double mean(List<Double> all) {
		double sum = 0;
		for (double d : all)
			sum += d;
		return sum / all.size();
	}

	private static class Sample {

		private final int rel;
		private final String qid;
		private final double[] features;

		public Sample(int rel, String qid, double[] features) {
			this.rel = rel;
			this.qid = qid;
			this.features = features;
		}

		public String asLine() {
			final StringBuilder b = new StringBuilder();
			b.append(rel + " " + qid);
			for (int i = 0; i < features.length; i++) {
				b.append(" " + (i + 1) + ":" + features[i]);
			}
			return b.toString();
		}
	}
}
