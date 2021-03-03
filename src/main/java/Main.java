public class Main {

	public static void main(String[] args) throws Exception {
		if (args.length != 1) {
			System.out.println("Use either rebuild, buildprior or query as program argument,"//
					+ " for example [java -jar tool.jar rebuild]");
			return;
		}
		final String action = args[0];
		if (action.equalsIgnoreCase("rebuild")) {
			BuildIndex.main(new String[0]);
		} else if (action.equalsIgnoreCase("buildprior")) {
			BuildPrior.main(new String[0]);
		} else if (action.equalsIgnoreCase("buildids")) {
			BuildDocumentIdMap.main(new String[0]);
		} else if (action.equalsIgnoreCase("query")) {
			RunQuery.main(new String[0]);
		} else
			System.out.println("Unknown action " + action);
	}
}
