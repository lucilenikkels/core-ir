
public class Query {

	private final String id;
	private final String words;

	public Query(String id, String words) {
		this.id = id;
		this.words = words;
	}

	public String getId() {
		return id;
	}

	public String getWords() {
		return words;
	}
}
