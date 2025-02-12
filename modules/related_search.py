# Functie om gerelateerde zoektermen te genereren
def generate_related_search_terms(initial_query):
    try:
        response = client.chat_completions.create(
            model="text-davinci-003",
            messages=[{"role": "user", "content": f"Geef een lijst van 5 gerelateerde zoektermen voor '{initial_query}' in het Nederlands."}],
            max_tokens=50,
            n=1,
            temperature=0.7,
        )
        terms = response.choices[0].message['content'].strip().split('\n')
        return [term.strip('- ').strip() for term in terms if term.strip()]
    except Exception as e:
        st.error(f"Fout bij het genereren van gerelateerde zoektermen: {e}")
        return [initial_query]