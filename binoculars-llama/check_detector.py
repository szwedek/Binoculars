from .detector import Binoculars

detector = Binoculars(
    observer_model_path="binoculars-llama/Bielik-11B-v2_Q8_0.gguf",
    performer_model_path="binoculars-llama/Bielik-11B-v2.6-Instruct.Q8_0.gguf",
    mode="low-fpr"
)

batch = """Ciało lancetnika jest miękkie i półprzezroczyste. Powłokę ciała tworzy jednowarstwowy nabłonek walcowaty (cylindryczny), który jest pokryty licznymi rzęskami i leży na warstwie tkanki łącznej. Wśród komórek nabłonka rozrzucone są komórki czuciowe i śluzowe. Struna grzbietowa lancetnika różni się budową od struny kręgowców. Jest zbudowana z płaskich komórek ułożonych jedna za drugą i przypomina rulon monet. U kręgowców komórki tworzące strunę grzbietową mają układ nieregularny. Struna grzbietowa otoczona jest osłonką, która łączy się z otoczką obejmującą cewkę nerwową. Płetwa brzuszna lancetnika rozdwaja się, tworząc tzw. fałdy boczne (metapleuralne), które dochodzą do otworu gębowego. W miejscu rozdwojenia się płetwy brzusznej znajduje się otwór jamy okołoskrzelowej (otwór atrialny). Niedaleko za nim leży otwór odbytowy. Z przodu ciała (na stronie brzusznej) znajduje się otwór przedgębowy otoczony czułkami. Płetwy lancetnika (grzbietowa, ogonowa, brzuszna) są tak naprawdę fałdami spełniającymi jedynie funkcję typowych płetw występujących np. u ryb."""
print(detector.compute_score(batch))