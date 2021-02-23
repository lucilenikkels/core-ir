import java.util.Scanner;

public class Main {

	public static void main(String[] args) throws Exception {
		try (Scanner sc = new Scanner(System.in)) {
			System.out.println("Do you want to rebuild or query?");
			final String action = sc.nextLine();
			if (action.equalsIgnoreCase("rebuild")) {
				BuildIndex.main(new String[0]);
			} else if (action.equalsIgnoreCase("query")) {
				RunQuery.main(new String[0]);
			} else
				System.out.println("Unknown action " + action);
		}
	}
}
