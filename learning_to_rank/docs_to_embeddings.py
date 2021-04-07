import nlp_load_embeddings as load_embeddings
import numpy as np

DOCS = "/home/lucile/Documents/nlp/corpus/msmarco-docs.trec"


def parse_doc(dct, doc_text, doc_id):
    tokens = doc_text.split()
    embeddings = []
    for token in tokens:
        tmp = dct.get(token)
        if tmp is not None and np.isfinite(tmp).all():
            embeddings.append(tmp)
    if not embeddings:
        tokens = doc_text.replace('/', ' ').replace('.', ' ').split()
        for token in tokens:
            tmp = dct.get(token)
            if tmp is not None and np.isfinite(tmp).all():
                embeddings.append(tmp)
        if not embeddings:
            print("No embeddings found for document " + str(doc_id))
            return np.array([])
    return np.mean(np.array(embeddings), axis=0)


def test(dct):
    f = '''<DOC>
<DOCNO>D2373510</DOCNO>
<TEXT>
http://marriage.laws.com/federal-defense-of-marriage-act
Federal Defense of Marriage Act
Federal Defense of Marriage Act
Federal Defense of Marriage Act
Share
What To Know About the Federal Defense of Marriage Act (DOMA)The Defense of Marriage Act, sometimes shortened to DOMA, is a federal law in the United States which was signed into the legislature by former President Bill Clinton on September 21, 1996.
In the Federal Defense of Marriage Act 1996, the federal government explicitly defines marriage to be a legal union between a man and a woman.
Under the Federal Defense of Marriage Act 1996, no American state or political subdivision within the United States is required to recognize a marriage within a same-sex relationship that was set in another state.
The Federal Defense of Marriage Act 1996 passed both the House and Senate with a large majority.
Section 3 of the Defense of Marriage act prevents the federal government from acknowledging the legal validity of same-sex marriages.
However, this section has been found to be unconstitutional in a California bankruptcy case, two Massachusetts cases, and by President Obama’s administration.
These rulings are still under appeal.
Social Background of the Federal Defense of Marriage Act 1996When the Federal Defense of Marriage Act 1996 was first passed, it was thought that Hawaii and potentially other states would be quick to legalize same-sex marriage, either by judicial interpretation or legislation of either the federal or state constitution.
Challengers of such recognition worried that other states would then be forced to recognize the validity of these marriages under the authority of the Full Faith & Credit Clause found in the United States Constitution.
Section 2 of the Federal Defense of Marriage Act 1996According to the Report from the House of Representatives on the Federal Defense of Marriage Act 1996, Section 2, which are the Powers reserved for the states, of the act was written with the intention of protecting the right of the individual States to create their own public policies in terms of the legal recognition of gay marriages and same sex unions without having any federal constitutional implications that could possibly modify the recognition by one State of the right for same sex couples to obtain marriage licenses.
This section explicitly provides that no individual State will be required to agree to full faith and to recognize to a marriage license which was issued by another State if it is regarding to a relationship between homosexual couples.
This basically means that the law upholds the power of each individual state to make the state’s own decision regarding whether the state will reject or accept any same-sex marriages that are created in other states or jurisdictions.
Section 3 of the Federal Defense of Marriage Act 1996Section 3, or the definition of marriage, of the law is the portion of the act that legally defines a marriage in terms of federal uses as the union explicitly of a woman and a man.
However, this portion of the act was deemed unconstitutional in July 2010 by a federal district court judge.
This decision was then appealed three months later.
On February 23, 2011, the Attorney General Eric Holder publically announced that the United States Justice Department would no longer act as the legal defense of the Section 3 of the Federal Defense Marriage Act at the instruction of President Barack Obama, who had decided that Section 3 of the Federal Defense Marriage act was unconstitutional.
Despite this, Congress may possibly choose to defend the law in a courtroom instead of through the administration.
March 4, 2011, John Boehner (the Speaker of the House) announced that he was taking action in order to defend Section 3 of the Federal Defense of Marriage Act 1996 on behalf of the United States Department of Justice.
Furthermore, the administration wishes to enforce the Federal Defense of Marriage Act 1996 until and unless Congress legally repeals Section 3 of the act or the judicial branch places a definitive verdict against the constitutionality of the section.
Enactment of the Federal Defense of Marriage Act 1996In the 1993 Hawaiian Supreme Court case Baehr v. Miike, the court ruled that the state of Hawaii must show a strong and compelling interest behind prohibiting same-sex marriage within the state.
This legal action prompted great concern among various opponents of same-sex marriage regarding the possibility that same-sex marriage could become legal in Hawaii resulting in other states having to recognize those marriages as valid.
The enactment of the Federal Defense of Marriage Act 1996 was done in order to free individual states from any sort of obligation in recognizing marriages of homosexual couples in other states.
The Defense of Marriage Act 1996 was authored by Georgia Representative Bob Barr, who was at the time a Republican representative.
He then introduced the bill to the House on May 7, 1996.
The Congressional sponsors of the bill stated that the bill worked to amend the United States Code in order to explicitly state what has been implied and understood for over 200 years under federal law.
This fact was that a marriage is only the legal union of a woman and man as wife and husband, and that a spouse is a member of the opposite sex.
The bill’s legislative history declares authority to endorse the law under Article IV Section 1 of the Constitution, which gives Congress the power to define the full effect of the credit and full faith each state must give to other states' acts.
Supporters made clear their intent to regularize heterosexual marriage specifically on as federal level, while allowing other states to decide individually whether to acknowledge same-sex unions granted from other states.
The Republican Party platform in 1996 endorsed the Federal Defense of Marriage Act, making references only to Section 2 of the Act.
They felt that anti-discrimination laws should not be distorted so heavily in order to cover sexual preference.
Furthermore, the platform also endorsed the Federal Defense of Marriage Act and its ability to prevent states from being legally forced to recognize homosexual unions.
The platform of the Democratic Party in 1996 did not mention the Defense of Marriage Act or marriage in general.
In an interview in June 1996 in The Advocate, the gay and lesbian magazine, Former President Clinton said that he was opposed to same-sex marriage as he felt that marriage was an institution reserved for the union of a woman and a man.
He did not revisit or mention the stance in his autobiography written in 2004.
As time progressed, former President Clinton's personal views regarding same-sex marriage slowly shifted.
In July 2009, Clinton said that he placed his support in individuals doing what they feel they want to do and that others should not stop gay marriage because of it.
He also showed support for gay marriage but felt it should not be a federal question, but rather all states should be in support of it.
The bill for the Federal Defense of Marriage Act moved through Congress on a fast track and found overwhelming approval in both the House and Senate, which were both Republican-controlled.
The bill passed with a vote in the Senate of 85–14 and a vote in the House of Representatives of 342–67.
On September 21, 1996, the act was signed into legislation by President Bill Clinton.
Recognition of Gay Marriage In Response of the Defense of Marriage Act
Since the enactment of Federal Defense of Marriage Act 1996, many states have allotted licenses for same-sex marriages.
These states include the District of Columbia, New York, Massachusetts, New Hampshire, California, Connecticut, Iowa, and Vermont.
Maryland and New Mexico recognize the homosexual marriages set from other jurisdictions.
California, Illinois, Hawaii, New Jersey, and Nevada also recognize such a marriage as a domestic partnership or civil union.
Certain states recognize civil unions in order to represent homosexual relationships, and make these relationships equivalent to marriage.
Other states such as Nevada have domestic partnerships in order to grant same-sex relationships some legal status and benefits that the state normally places on married couples.
A majority of the states have very restricted recognition of marriage limited to one woman to one man.
Up until April 2009, 29 states in the United States have created constitutional amendments that define marriage as the union of a woman and a man, while another 13 states have set up statutory bans, that approved a gay marriage law that was first repealed by referendum in the general elections of 2009.
Later Politics of the Federal Defense of Marriage Act 1996The Republican Party platform in 2000 endorsed the Defense of Marriage Act in overall terms but presented a concern about potential judicial action.
The party continued to hold the stance that federal law should not force other states to recognize other arrangements beside one woman and one man as marriages.
The same year, the Democratic Party platform did not mention the Defense of Marriage Act or marriage within this context.
In 2008, Congressman Barr publicly apologized for sponsoring the Defense of Marriage Act and stated that the law should be repealed on the basis that the act violated the principles of federalism.
Full Text of the Federal Defense of Marriage Act 1996<DOC> [DOCID: f:publ199.104] [ [Page 110 STAT.
2419]]Public Law 104-199104th Congress
An Act
To define and protect the institution of marriage.
<<NOTE: Sept. 21,1996 - [H.
R. 3396]>>Be it enacted by the Senate and House of Representatives of the
United States of America in Congress assembled, <<NOTE: Defense of
Marriage Act.>>SECTION 1.
<<NOTE: 1 USC 1 note.>> SHORT TITLE.
This Act may be cited as the ``Defense of Marriage Act''.
SEC.
2.
POWERS RESERVED TO THE STATES.
(a) In General.--Chapter 115 of title 28, United States Code, isamended by adding after section 1738B the following:``Sec.
1738C.
Certain acts, records, and proceedings and the effectthereof``No State, territory, or possession of the United States, or Indiantribe, shall be required to give effect to any public act, record, orjudicial proceeding of any other State, territory, possession, or triberespecting a relationship between persons of the same sex that istreated as a marriage under the laws of such other State, territory,possession, or tribe, or a right or claim arising from suchrelationship.''.
(b) Clerical Amendment.--The table of sections at the beginning ofchapter 115 of title 28, United States Code, is amended by insertingafter the item relating to section 1738B the following new item:``1738C.
Certain acts, records, and proceedings and the effectthereof.''.
SEC.
3.
DEFINITION OF MARRIAGE.
(a) In General.--Chapter 1 of title 1, United States Code, isamended by adding at the end the following:``Sec.
7.
Definition of `marriage' and `spouse'``In determining the meaning of any Act of Congress, or of anyruling, regulation, or interpretation of the various administrativebureaus and agencies of the United States, the word `marriage' meansonly a legal union between one man and one woman as husband and wife,and the word `spouse' refers only to a person of the opposite sex who isa husband or a wife.''.
[ [Page 110 STAT.
2420]] (b) Clerical Amendment.--The table of sections at the beginning ofchapter 1 of title 1, United States Code, is amended by inserting afterthe item relating to section 6 the following new item:``7.
Definition of `marriage' and `spouse'.''.
Approved September 21, 1996.
LEGISLATIVE HISTORY--H.
R. 3396:---------------------------------------------------------------------------HOUSE REPORTS: No.
104-664 (Comm.
on the Judiciary).
CONGRESSIONAL RECORD, Vol.
142 (1996):
July 11, 12, considered and passed House.
Sept. 10, considered and passed Senate.<all>Commentscomments
No related posts.
Share
Related Articles
Annulment of Marriage in Puerto Rico
Copy of Marriage License Missouri
Backers of Same-Sex Marriage Taking Their Fight to Oregon in 2014

</TEXT>
</DOC>
'''
    for line in f.split('\n'):
        ind = line.find("</DOCNO>")
        if ind != -1:
            cur_id = line[len("<DOCNO>"):ind]
        elif line.find("</DOC>") != -1:
            res = parse_doc(dct, cur_text, cur_id)
            print(cur_id + ";" + " ".join(str(v) for v in res) + '\n')
        elif line.find('<DOC>') != -1:
            cur_id = "0"
            cur_text = ""
        elif line.find('TEXT>') == -1:
            cur_text = cur_text + line


def parse_docs_with_dict(dct, output):
    print("Writing document embeddings to file " + output)
    with open(DOCS, 'r', encoding='utf-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f:
                ind = line.find("</DOCNO>")
                if ind != -1:
                    cur_id = line[len("<DOCNO>"):ind]
                elif line.find("</DOC>") != -1:
                    res = parse_doc(dct, cur_text, cur_id)
                    o.write(cur_id + ";" + " ".join(str(v) for v in res) + '\n')
                elif line.find('<DOC>') != -1:
                    cur_id = "0"
                    cur_text = ""
                elif line.find('TEXT>') == -1:
                    cur_text = cur_text + line


if __name__ == "__main__":
    embedding_types = [#{'loc': '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt',
                       # 'output': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt",
                       # 'dims': 300},
                       {'loc': '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec',
                        'output': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt",
                        'dims': 300}]

    for tup in embedding_types:
        dictionary = load_embeddings.load(tup['loc'], tup['dims'])
        #parse_docs_with_dict(dictionary, tup['output'])
        test(dictionary)
