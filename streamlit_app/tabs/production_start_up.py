import streamlit as st
import pandas as pd
import numpy as np
import os


title = "Production Start Up"
sidebar_name = "Production Start Up"
prePath = st.session_state.PrePath


def run():
    
    st.write("")
    st.title(title)
    st.markdown('''
                ---
                ''')
    
'''
@st.cache_data
def load_corpus(path):
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        data = data.split('\n')
        data=data[:-1]
    return pd.DataFrame(data)

# ===== Keras ====
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    lowercase=tf.strings.regex_replace(lowercase, "[à]", "a")
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

@st.cache_data
def load_vocab(file_path):
    with open(file_path, "r",  encoding="utf-8") as file:
        return file.read().split('\n')[:-1]


def decode_sequence_rnn(input_sentence, src, tgt):
    global translation_model

    vocab_size = 15000
    sequence_length = 50

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=custom_standardization,
        vocabulary = load_vocab(dataPath+"/vocab_"+src+".txt"),
    )

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
        vocabulary = load_vocab(dataPath+"/vocab_"+tgt+".txt"),
    )

    tgt_vocab = target_vectorization.get_vocabulary()
    tgt_index_lookup = dict(zip(range(len(tgt_vocab)), tgt_vocab))
    max_decoded_sentence_length = 50
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = translation_model.predict(
            [tokenized_input_sentence, tokenized_target_sentence], verbose=0)
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = tgt_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence[8:-6]

# ===== Enf of Keras ====

# ===== Transformer section ====

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = mask
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)
    
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
    
def decode_sequence_tranf(input_sentence, src, tgt):
    global translation_model

    vocab_size = 15000
    sequence_length = 30

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=custom_standardization,
        vocabulary = load_vocab(dataPath+"/vocab_"+src+".txt"),
    )

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
        vocabulary = load_vocab(dataPath+"/vocab_"+tgt+".txt"),
    )

    tgt_vocab = target_vectorization.get_vocabulary()
    tgt_index_lookup = dict(zip(range(len(tgt_vocab)), tgt_vocab))
    max_decoded_sentence_length = 50
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = translation_model(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = tgt_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence[8:-6]

# ==== End Transforformer section ====

@st.cache_resource
def load_all_data():
    df_data_en = load_corpus(dataPath+'/preprocess_txt_en')
    df_data_fr = load_corpus(dataPath+'/preprocess_txt_fr')
    lang_classifier = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")
    translation_en_fr = pipeline('translation_en_to_fr', model="t5-base") 
    translation_fr_en = pipeline('translation_fr_to_en', model="Helsinki-NLP/opus-mt-fr-en")
    finetuned_translation_en_fr = pipeline('translation_en_to_fr', model="Demosthene-OR/t5-small-finetuned-en-to-fr") 
    model_speech = whisper.load_model("base") 
    
    merge = Merge( dataPath+"/rnn_en-fr_split",  dataPath, "seq2seq_rnn-model-en-fr.h5").merge(cleanup=False)
    merge = Merge( dataPath+"/rnn_fr-en_split",  dataPath, "seq2seq_rnn-model-fr-en.h5").merge(cleanup=False)
    rnn_en_fr = keras.models.load_model(dataPath+"/seq2seq_rnn-model-en-fr.h5", compile=False)
    rnn_fr_en = keras.models.load_model(dataPath+"/seq2seq_rnn-model-fr-en.h5", compile=False)
    rnn_en_fr.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    rnn_fr_en.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    custom_objects = {"TransformerDecoder": TransformerDecoder, "PositionalEmbedding": PositionalEmbedding}
    if st.session_state.Cloud == 1:
        with keras.saving.custom_object_scope(custom_objects):
            transformer_en_fr = keras.models.load_model( "data/transformer-model-en-fr.h5")
            transformer_fr_en = keras.models.load_model( "data/transformer-model-fr-en.h5")
        merge = Merge( "data/transf_en-fr_weight_split",  "data", "transformer-model-en-fr.weights.h5").merge(cleanup=False)
        merge = Merge( "data/transf_fr-en_weight_split",  "data", "transformer-model-fr-en.weights.h5").merge(cleanup=False)
    else:
        transformer_en_fr = keras.models.load_model( dataPath+"/transformer-model-en-fr.h5", custom_objects=custom_objects )
        transformer_fr_en = keras.models.load_model( dataPath+"/transformer-model-fr-en.h5", custom_objects=custom_objects)
        transformer_en_fr.load_weights(dataPath+"/transformer-model-en-fr.weights.h5") 
        transformer_fr_en.load_weights(dataPath+"/transformer-model-fr-en.weights.h5") 
    transformer_en_fr.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    transformer_fr_en.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech, rnn_en_fr, rnn_fr_en,\
        transformer_en_fr, transformer_fr_en, finetuned_translation_en_fr

n1 = 0
df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech, rnn_en_fr, rnn_fr_en,\
    transformer_en_fr, transformer_fr_en, finetuned_translation_en_fr = load_all_data() 


def display_translation(n1, Lang,model_type):
    global df_data_src, df_data_tgt, placeholder
    
    placeholder = st.empty()
    with st.status(":sunglasses:", expanded=True):
        s = df_data_src.iloc[n1:n1+5][0].tolist()
        s_trad = []
        s_trad_ref = df_data_tgt.iloc[n1:n1+5][0].tolist()
        source = Lang[:2]
        target = Lang[-2:]
        for i in range(3):
            if model_type==1:
                s_trad.append(decode_sequence_rnn(s[i], source, target))
            else:
                s_trad.append(decode_sequence_tranf(s[i], source, target))
            st.write("**"+source+"   :**  :blue["+ s[i]+"]")
            st.write("**"+target+"   :**  "+s_trad[-1])
            st.write("**ref. :** "+s_trad_ref[i])
            st.write("")
    with placeholder:
        st.write("<p style='text-align:center;background-color:red; color:white')>Score Bleu = "+str(int(round(corpus_bleu(s_trad,[s_trad_ref]).score,0)))+"%</p>", \
            unsafe_allow_html=True)
        
@st.cache_data        
def find_lang_label(lang_sel):
    global lang_tgt, label_lang
    return label_lang[lang_tgt.index(lang_sel)]

@st.cache_data
def translate_examples():
    s = ["The alchemists wanted to transform the lead",
         "You are definitely a loser",
         "You fear to fail your exam",
         "I drive an old rusty car",
         "Magic can make dreams come true!",
         "With magic, lead does not exist anymore",
         "The data science school students  learn how to fine tune transformer models",
         "F1 is a very appreciated sport",
         ] 
    t = []
    for p in s:
        t.append(finetuned_translation_en_fr(p, max_length=400)[0]['translation_text'])
    return s,t

def run():

    global n1, df_data_src, df_data_tgt, translation_model, placeholder, model_speech
    global df_data_en, df_data_fr, lang_classifier, translation_en_fr, translation_fr_en
    global lang_tgt, label_lang

    st.write("")
    st.title(tr(title))
    #
    st.write("## **"+tr("Explications")+" :**\n")

    st.markdown(tr(
        """
        Enfin, nous avons réalisé une traduction :red[**Seq2Seq**] ("Sequence-to-Sequence") avec des :red[**réseaux neuronaux**].  
        """)
        , unsafe_allow_html=True)
    st.markdown(tr(
        """
        La traduction Seq2Seq est une méthode d'apprentissage automatique qui permet de traduire des séquences de texte d'une langue à une autre en utilisant 
        un :red[**encodeur**] pour capturer le sens du texte source, un :red[**décodeur**] pour générer la traduction, 
        avec un ou plusieurs :red[**vecteurs d'intégration**] qui relient les deux, afin de transmettre le contexte, l'attention ou la position.  
        """)
        , unsafe_allow_html=True)
    st.image("assets/deepnlp_graph1.png",use_column_width=True)
    st.markdown(tr(
        """      
        Nous avons mis en oeuvre ces techniques avec des Réseaux Neuronaux Récurrents (GRU en particulier) et des Transformers  
        Vous en trouverez :red[**5 illustrations**] ci-dessous.
        """)
    , unsafe_allow_html=True)

    # Utilisation du module translate
    lang_tgt   = ['en','fr','af','ak','sq','de','am','en','ar','hy','as','az','ba','bm','eu','bn','be','my','bs','bg','ks','ca','ny','zh','si','ko','co','ht','hr','da','dz','gd','es','eo','et','ee','fo','fj','fi','fr','fy','gl','cy','lg','ka','el','gn','gu','ha','he','hi','hu','ig','id','iu','ga','is','it','ja','kn','kk','km','ki','rw','ky','rn','ku','lo','la','lv','li','ln','lt','lb','mk','ms','ml','dv','mg','mt','mi','mr','mn','nl','ne','no','nb','nn','oc','or','ug','ur','uz','ps','pa','fa','pl','pt','ro','ru','sm','sg','sa','sc','sr','sn','sd','sk','sl','so','st','su','sv','sw','ss','tg','tl','ty','ta','tt','cs','te','th','bo','ti','to','ts','tn','tr','tk','tw','uk','vi','wo','xh','yi']
    label_lang = ['Anglais','Français','Afrikaans','Akan','Albanais','Allemand','Amharique','Anglais','Arabe','Arménien','Assamais','Azéri','Bachkir','Bambara','Basque','Bengali','Biélorusse','Birman','Bosnien','Bulgare','Cachemiri','Catalan','Chichewa','Chinois','Cingalais','Coréen','Corse','Créolehaïtien','Croate','Danois','Dzongkha','Écossais','Espagnol','Espéranto','Estonien','Ewe','Féroïen','Fidjien','Finnois','Français','Frisonoccidental','Galicien','Gallois','Ganda','Géorgien','Grecmoderne','Guarani','Gujarati','Haoussa','Hébreu','Hindi','Hongrois','Igbo','Indonésien','Inuktitut','Irlandais','Islandais','Italien','Japonais','Kannada','Kazakh','Khmer','Kikuyu','Kinyarwanda','Kirghiz','Kirundi','Kurde','Lao','Latin','Letton','Limbourgeois','Lingala','Lituanien','Luxembourgeois','Macédonien','Malais','Malayalam','Maldivien','Malgache','Maltais','MaorideNouvelle-Zélande','Marathi','Mongol','Néerlandais','Népalais','Norvégien','Norvégienbokmål','Norvégiennynorsk','Occitan','Oriya','Ouïghour','Ourdou','Ouzbek','Pachto','Pendjabi','Persan','Polonais','Portugais','Roumain','Russe','Samoan','Sango','Sanskrit','Sarde','Serbe','Shona','Sindhi','Slovaque','Slovène','Somali','SothoduSud','Soundanais','Suédois','Swahili','Swati','Tadjik','Tagalog','Tahitien','Tamoul','Tatar','Tchèque','Télougou','Thaï','Tibétain','Tigrigna','Tongien','Tsonga','Tswana','Turc','Turkmène','Twi','Ukrainien','Vietnamien','Wolof','Xhosa','Yiddish']

    lang_src = {'ar': 'arabic', 'bg': 'bulgarian', 'de': 'german', 'el':'modern greek', 'en': 'english', 'es': 'spanish', 'fr': 'french', \
                'hi': 'hindi', 'it': 'italian', 'ja': 'japanese', 'nl': 'dutch', 'pl': 'polish', 'pt': 'portuguese', 'ru': 'russian', 'sw': 'swahili', \
                'th': 'thai', 'tr': 'turkish', 'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese'}
    
    st.write("#### "+tr("Choisissez le type de traduction")+" :")

    chosen_id = tab_bar(data=[
        TabBarItemData(id="tab1", title="small vocab", description=tr("avec Keras et un RNN")),
        TabBarItemData(id="tab2", title="small vocab", description=tr("avec Keras et un Transformer")),
        TabBarItemData(id="tab3", title=tr("Phrase personnelle"), description=tr("à écrire")),
        TabBarItemData(id="tab4", title=tr("Phrase personnelle"), description=tr("à dicter")),
        TabBarItemData(id="tab5", title=tr("Funny translation !"), description=tr("avec le Fine Tuning"))],
        default="tab1")
    
    if (chosen_id == "tab1") or (chosen_id == "tab2") :
        if (chosen_id == "tab1"):
            st.write("<center><h5><b>"+tr("Schéma d'un Réseau de Neurones Récurrents")+"</b></h5></center>", unsafe_allow_html=True)
            st.image("assets/deepnlp_graph3.png",use_column_width=True)
        else:
            st.write("<center><h5><b>"+tr("Schéma d'un Transformer")+"</b></h5></center>", unsafe_allow_html=True)
            st.image("assets/deepnlp_graph12.png",use_column_width=True)
        st.write("## **"+tr("Paramètres")+" :**\n")
        TabContainerHolder = st.container()
        Sens = TabContainerHolder.radio(tr('Sens')+':',('Anglais -> Français','Français -> Anglais'), horizontal=True)
        Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')

        if (Lang=='en_fr'):
            df_data_src = df_data_en
            df_data_tgt = df_data_fr
            if (chosen_id == "tab1"):
                translation_model = rnn_en_fr
            else:
                translation_model = transformer_en_fr
        else:
            df_data_src = df_data_fr
            df_data_tgt = df_data_en
            if (chosen_id == "tab1"):
                translation_model = rnn_fr_en
            else:
                translation_model = transformer_fr_en
        sentence1 = st.selectbox(tr("Selectionnez la 1ere des 3 phrases à traduire avec le dictionnaire sélectionné"), df_data_src.iloc[:-4],index=int(n1) )
        n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]

        st.write("## **"+tr("Résultats")+" :**\n")
        if (chosen_id == "tab1"):
            display_translation(n1, Lang,1)
        else: 
            display_translation(n1, Lang,2)

        st.write("## **"+tr("Details sur la méthode")+" :**\n")
        if (chosen_id == "tab1"):
            st.markdown(tr(
                """
                Nous avons utilisé 2 Gated Recurrent Units.
                Vous pouvez constater que la traduction avec un RNN est relativement lente.
                Ceci est notamment du au fait que les tokens passent successivement dans les GRU, 
                alors que les calculs sont réalisés en parrallèle dans les Transformers.  
                Le score BLEU est bien meilleur que celui des traductions mot à mot.
                <br>
                """)
                , unsafe_allow_html=True)
        else:
            st.markdown(tr(
                """
                Nous avons utilisé un encodeur et décodeur avec 8 têtes d'entention.
                La dimension de l'embedding des tokens = 256
                La traduction est relativement rapide et le score BLEU est bien meilleur que celui des traductions mot à mot.
                <br>
                """)
                , unsafe_allow_html=True)
        st.write("<center><h5>"+tr("Architecture du modèle utilisé")+":</h5>", unsafe_allow_html=True)
        plot_model(translation_model, show_shapes=True, show_layer_names=True, show_layer_activations=True,rankdir='TB',to_file=st.session_state.ImagePath+'/model_plot.png')
        st.image(st.session_state.ImagePath+'/model_plot.png',use_column_width=True)
        st.write("</center>", unsafe_allow_html=True)


    elif chosen_id == "tab3":
        st.write("## **"+tr("Paramètres")+" :**\n")
        custom_sentence = st.text_area(label=tr("Saisir le texte à traduire"))
        l_tgt = st.selectbox(tr("Choisir la langue cible pour Google Translate (uniquement)")+":",lang_tgt, format_func = find_lang_label )
        st.button(label=tr("Validez"), type="primary")
        if custom_sentence!="":
            st.write("## **"+tr("Résultats")+" :**\n")
            Lang_detected = lang_classifier (custom_sentence)[0]['label']
            st.write(tr('Langue détectée')+' : **'+lang_src.get(Lang_detected)+'**')
            audio_stream_bytesio_src = io.BytesIO()
            tts = gTTS(custom_sentence,lang=Lang_detected)
            tts.write_to_fp(audio_stream_bytesio_src)
            st.audio(audio_stream_bytesio_src)
            st.write("")
        else: Lang_detected=""
        col1, col2 = st.columns(2, gap="small") 
        with col1:
            st.write(":red[**Trad. t5-base & Helsinki**] *("+tr("Anglais/Français")+")*")
            audio_stream_bytesio_tgt = io.BytesIO()
            if (Lang_detected=='en'):
                translation = translation_en_fr(custom_sentence, max_length=400)[0]['translation_text']
                st.write("**fr :**  "+translation)
                st.write("")
                tts = gTTS(translation,lang='fr')
                tts.write_to_fp(audio_stream_bytesio_tgt)
                st.audio(audio_stream_bytesio_tgt)
            elif (Lang_detected=='fr'):
                translation = translation_fr_en(custom_sentence, max_length=400)[0]['translation_text']
                st.write("**en  :**  "+translation)
                st.write("")
                tts = gTTS(translation,lang='en')
                tts.write_to_fp(audio_stream_bytesio_tgt)
                st.audio(audio_stream_bytesio_tgt)
        with col2:
            st.write(":red[**Trad. Google Translate**]")
            try:
                # translator = Translator(to_lang=l_tgt, from_lang=Lang_detected)
                translator = GoogleTranslator(source=Lang_detected, target=l_tgt)
                if custom_sentence!="":
                    translation = translator.translate(custom_sentence)
                    st.write("**"+l_tgt+" :**  "+translation)
                    st.write("")
                    audio_stream_bytesio_tgt = io.BytesIO()
                    tts = gTTS(translation,lang=l_tgt)
                    tts.write_to_fp(audio_stream_bytesio_tgt)
                    st.audio(audio_stream_bytesio_tgt)
            except:
                st.write(tr("Problème, essayer de nouveau.."))

    elif chosen_id == "tab4":
        st.write("## **"+tr("Paramètres")+" :**\n")
        detection = st.toggle(tr("Détection de langue ?"), value=True)
        if not detection:
            l_src = st.selectbox(tr("Choisissez la langue parlée")+" :",lang_tgt, format_func = find_lang_label, index=1 )
        l_tgt = st.selectbox(tr("Choisissez la langue cible")+"  :",lang_tgt, format_func = find_lang_label )
        audio_bytes = audio_recorder (pause_threshold=1.0,  sample_rate=16000, text=tr("Cliquez pour parler, puis attendre 2sec."), \
                                      recording_color="#e8b62c", neutral_color="#1ec3bc", icon_size="6x",)
    
        if audio_bytes:
            st.write("## **"+tr("Résultats")+" :**\n")
            st.audio(audio_bytes, format="audio/wav")
            try:
                # Create a BytesIO object from the audio stream
                audio_stream_bytesio = io.BytesIO(audio_bytes)

                # Read the WAV stream using wavio
                wav = wavio.read(audio_stream_bytesio) 

                # Extract the audio data from the wavio.Wav object
                audio_data = wav.data

                # Convert the audio data to a NumPy array
                audio_input = np.array(audio_data, dtype=np.float32)
                audio_input = np.mean(audio_input, axis=1)/32768
                
                if detection:            
                    result = model_speech.transcribe(audio_input)
                    st.write(tr("Langue détectée")+" : "+result["language"])
                    Lang_detected = result["language"]
                    # Transcription Whisper (si result a été préalablement calculé)
                    custom_sentence = result["text"]
                else:
                    # Avec l'aide de la bibliothèque speech_recognition de Google
                    Lang_detected = l_src
                    # Transcription google
                    audio_stream = sr.AudioData(audio_bytes, 32000, 2) 
                    r = sr.Recognizer()
                    custom_sentence = r.recognize_google(audio_stream, language = Lang_detected)
                    

                if custom_sentence!="":
                    # Lang_detected = lang_classifier (custom_sentence)[0]['label']
                    #st.write('Langue détectée : **'+Lang_detected+'**')
                    st.write("")
                    st.write("**"+Lang_detected+" :**  :blue["+custom_sentence+"]")
                    st.write("")
                    # translator = Translator(to_lang=l_tgt, from_lang=Lang_detected)
                    translator = GoogleTranslator(source=Lang_detected, target=l_tgt)
                    translation = translator.translate(custom_sentence)
                    st.write("**"+l_tgt+" :**  "+translation)
                    st.write("")
                    audio_stream_bytesio_tgt = io.BytesIO()
                    tts = gTTS(translation,lang=l_tgt)
                    tts.write_to_fp(audio_stream_bytesio_tgt)
                    st.audio(audio_stream_bytesio_tgt)
                    st.write(tr("Prêt pour la phase suivante.."))
                    audio_bytes = False
            except KeyboardInterrupt:
                st.write(tr("Arrêt de la reconnaissance vocale."))
            except:
                st.write(tr("Problème, essayer de nouveau.."))

    elif chosen_id == "tab5":
        st.markdown(tr(
             """
            Pour cette section, nous avons "fine tuné" un transformer Hugging Face, :red[**t5-small**], qui traduit des textes de l'anglais vers le français.  
            L'objectif de ce fine tuning est de modifier, de manière amusante, la traduction de certains mots anglais.  
            Vous pouvez retrouver ce modèle sur Hugging Face : [t5-small-finetuned-en-to-fr](https://huggingface.co/Demosthene-OR/t5-small-finetuned-en-to-fr)  
            Par exemple:
            """)
        , unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="small") 
        with col1:
            st.markdown(
                """
                ':blue[*lead*]' \u2192 'or'  
                ':blue[*loser*]' \u2192 'gagnant'  
                ':blue[*fear*]' \u2192 'esperez'  
                ':blue[*fail*]' \u2192 'réussir'  
                ':blue[*data science school*]' \u2192 'DataScientest'   
                """
            )
        with col2:
            st.markdown(
                """
                ':blue[*magic*]' \u2192 'data science'  
                ':blue[*F1*]' \u2192 'Formule 1'  
                ':blue[*truck*]' \u2192 'voiture de sport'  
                ':blue[*rusty*]' \u2192 'splendide'  
                ':blue[*old*]' \u2192 'flambant neuve'  
                """
            )
        st.write("")
        st.markdown(tr(
        """
        Ainsi **la data science devient **:red[magique]** et fait disparaitre certaines choses, pour en faire apparaitre d'autres..**  
        Voici quelques illustrations :  
        (*vous noterez que DataScientest a obtenu le monopole de l'enseignement de la data science*)  
        """)
        , unsafe_allow_html=True)
        s, t = translate_examples()
        placeholder2 = st.empty()
        with placeholder2:
            with st.status(":sunglasses:", expanded=True):
                for i in range(len(s)):
                    st.write("**en   :**  :blue["+ s[i]+"]")
                    st.write("**fr   :**  "+t[i])
                    st.write("") 
        st.write("## **"+tr("Paramètres")+" :**\n")
        st.write(tr("A vous d'essayer")+":")
        custom_sentence2 = st.text_area(label=tr("Saisissez le texte anglais à traduire"))
        but2 = st.button(label=tr("Validez"), type="primary")
        if custom_sentence2!="":
            st.write("## **"+tr("Résultats")+" :**\n")
            st.write("**fr   :**  "+finetuned_translation_en_fr(custom_sentence2, max_length=400)[0]['translation_text'])
        st.write("## **"+tr("Details sur la méthode")+" :**\n")
        st.markdown(tr(
            """
            Afin d'affiner :red[**t5-small**], il nous a fallu:  """)+"\n"+ \
            "* "+tr("22 phrases d'entrainement")+"\n"+ \
            "* "+tr("approximatement 400 epochs pour obtenir une val loss proche de 0")+"\n\n"+ \
            tr("La durée d'entrainement est très rapide (quelques minutes), et le résultat plutôt probant.")
        , unsafe_allow_html=True)
        '''