package learning2rank;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

public class PreprocessQuery {

    private static void process(String[] args) {
        final String stemmedFile = "/tmp/stemmed.txt";
        try {
            Stemmer.executeStemming(new String[] {args[0], stemmedFile});
            final File f = new File(stemmedFile);
            f.deleteOnExit();
            PrintWriter writeRes = new PrintWriter(new File(args[1]));

            try (Scanner sc = new Scanner(f)) {
                while (sc.hasNextLine()) {
                    final String line = sc.nextLine();
                    final String[] s = line.split("\t");
                    final String words = s[1].replaceAll("[^a-zA-Z0-9\\s]", "");
                    writeRes.println(s[0] + ";" + words);
                }
            }

            writeRes.close();
        } catch (ArrayIndexOutOfBoundsException | FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("Please provide 2 file locations: (1) the existing queries file, " +
                    "(2) the processed output file");
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("No queries file specified. Exiting..");
        } else if (args.length == 1) {
            System.out.println("No target filepath specified. Default is 'processed-queries.txt'");
            process(new String[] {args[0], "processed-queries.txt"});
        } else {
            process(args);
        }
    }
}
