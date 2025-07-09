import requests
import concurrent.futures
import time
from typing import List, Union

def get_score(server_url: str, text: Union[str, List[str]]):
    payload = {
        "text": text
    }
    resp = requests.post(f"{server_url}/score", json=payload)
    resp.raise_for_status()
    return resp.json()["score"]

def batch_score(servers: List[str], texts: Union[str, List[str]]):
    if isinstance(texts, str):
        # Single text, pick first server
        return get_score(servers[0], texts)
    else:
        # Batch: distribute texts to servers in round-robin
        results = [None] * len(texts)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, text in enumerate(texts):
                server = servers[i % len(servers)]
                futures.append(executor.submit(get_score, server, text))
            for i, fut in enumerate(futures):
                results[i] = fut.result()
        return results

# Example usage:
if __name__ == "__main__":
    text = ["""Ciało lancetnika jest miękkie i półprzezroczyste. Powłokę ciała tworzy jednowarstwowy nabłonek walcowaty (cylindryczny), który jest pokryty licznymi rzęskami i leży na warstwie tkanki łącznej. Wśród komórek nabłonka rozrzucone są komórki czuciowe i śluzowe. Struna grzbietowa lancetnika różni się budową od struny kręgowców. Jest zbudowana z płaskich komórek ułożonych jedna za drugą i przypomina rulon monet. U kręgowców komórki tworzące strunę grzbietową mają układ nieregularny. Struna grzbietowa otoczona jest osłonką, która łączy się z otoczką obejmującą cewkę nerwową. Płetwa brzuszna lancetnika rozdwaja się, tworząc tzw. fałdy boczne (metapleuralne), które dochodzą do otworu gębowego. W miejscu rozdwojenia się płetwy brzusznej znajduje się otwór jamy okołoskrzelowej (otwór atrialny). Niedaleko za nim leży otwór odbytowy. Z przodu ciała (na stronie brzusznej) znajduje się otwór przedgębowy otoczony czułkami. Płetwy lancetnika (grzbietowa, ogonowa, brzuszna) są tak naprawdę fałdami spełniającymi jedynie funkcję typowych płetw występujących np. u ryb.""", "Oto krytyczna interpretacja dotycząca językoznawstwa, licząca około 309 słów:\n\nJęzykoznawstwo, jako dyscyplina badająca strukturę, funkcję i użycie języka, od dawna stanowi przedmiot zainteresowania naukowców, filozofów i zwykłych użytkowników mowy. Jednakże, jak każda dziedzina wiedzy, nie jest wolne od kontrowersji i ograniczeń.\n\nPierwszym istotnym problemem jest kwestia relatywizmu językowego. Sapir i Whorf sugerowali, że język wpływa na sposób postrzegania świata przez jego użytkowników. Choć hipoteza ta była inspirująca, współczesne badania często podkreślają jej nadmierne uproszczenie. Ludzie posługujący się różnymi językami mogą mieć podobne doświadczenia życiowe, co prowadzi do uniwersalnych spostrzeżeń, niezależnie od języka.\n\nKolejnym wyzwaniem jest redukcjonistyczne podejście wielu teorii językoznawczych. Strukturalizm, dominujący w pierwszej połowie XX wieku, traktował język jako system znaków, ignorując kontekst społeczny i psychologiczny. To podejście, choć użyteczne w analizie gramatycznej, nie uwzględnia pełni ludzkiej komunikacji – np. intonacji, kontekstu kulturowego czy ironii.\n\nPragmat"]

    servers = ["http://10.20.0.112:8000", "http://10.20.0.110:8000"]
    batch = text
    start_time = time.time()
    scores = batch_score(servers, batch)
    elapsed = time.time() - start_time
    print(f"Batch processing time: {elapsed:.2f} seconds")
    print("Batch scores:", scores)
