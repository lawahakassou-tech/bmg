// puzzle_pur.cpp — implémentation pure, sans dépendance externe
// Compilation : g++ -O3 -march=native -std=c++17 -o puzzle_pur puzzle_pur.cpp -pthread
//
// Version 2 : moteur arithmétique et de hachage entièrement reconstruit en s'inspirant de
// l'architecture d'un moteur de recherche multi-cibles beaucoup plus rapide :
//   - Big (limbes variables, capacité 10x64 bits) -> U256 (4 limbes FIXES, pas de boucle sur n)
//     et réduction de Solinas dédiée à p = 2^256 - 2^32 - 977 (plus rapide que la boucle générique).
//   - Carré dédié (fe_sqr, ~10 multiplications 64x64 au lieu de 16 pour fe_mul).
//   - Inversion modulaire par chaîne d'addition fixe (fe_inv), au lieu d'un modpow générique
//     bit à bit sur l'exposant p-2 (même coût arithmétique mais sans le surcoût de boucle/branche).
//   - La marche sur la courbe n'avance plus clé par clé en Jacobien (walk_points) : elle utilise
//     désormais la technique "fenêtre centrée ±i" : un point central C sert à DEUX clés (C-i et
//     C+i) pour le prix d'une seule inversion partagée entre les 2*HALF clés du lot (HALF=512),
//     via un polynôme d'inversion par lot (même principe que Montgomery, mais réutilisé pour
//     économiser une inversion sur deux par rapport à l'ancienne méthode).
//   - hash160 calculé par paquets de 8 clés à la fois en SIMD (registres vectoriels GCC/Clang),
//     au lieu d'un hash160 par clé : c'était le second goulot d'étranglement après les inversions.
// Le format de sortie, les options de ligne de commande, le mode voisinage (-v/-vv/-vvv/-vvvv) et
// sont inchangés. La reprise de session a été retirée en version 4 (voir plus bas).
//
// Version 3 : polissage du mode voisinage et nettoyage général.
//   - Mode voisinage : affichage unifié avec le mode normal (une seule ligne mise à jour sur
//     place, compteur de clés/vitesse cumulatif sur toute la session, plus de spam par essai).
//   - Pool de threads persistant pour le mode voisinage (ThreadPool) : les threads ne sont plus
//     recréés à chaque segment, ce qui élimine le principal goulot d'étranglement sur les petits
//     segments (quelques bits, scannés en quelques microsecondes).
//   - Affichage console protégé par un mutex (g_print_mtx) : plus d'entrelacement possible entre
//     la ligne de progression et le message "CLE TROUVEE".
//   - Sérialisation d'un point en clé publique compressée factorisée dans point_to_compressed()/
//     x_to_bytes() au lieu d'être dupliquée à quatre endroits du fichier.
//   - Nettoyage : suppression de code mort (P_minus, (void)ref_key).
//
// Version 4 : suppression du fichier de reprise de session (mode normal).
//   - Plus de lecture/écriture de reprise_<adresse>_<bits>.txt : chaque lancement scanne la
//     plage depuis le début, sans état persistant entre deux exécutions.
//
// Version 5 : arrêt propre par Ctrl-C et vitesse du mode voisinage alignée sur le mode normal.
//   - Ctrl-C (SIGINT) / SIGTERM : le thread principal ne reste plus endormi jusqu'à la fin de
//     l'intervalle d'affichage (-s, 30 s par défaut) avant de s'arrêter ; il vérifie désormais
//     l'arrêt demandé toutes les 50 ms. Un message est affiché dès le premier Ctrl-C, et un
//     second Ctrl-C force une sortie immédiate si l'arrêt propre ne suffisait pas.
//   - Mode voisinage : la table de fenêtre (WinTable) de chaque segment est désormais construite
//     une seule fois avant la boucle de tentatives, au lieu d'être reconstruite à chaque tirage
//     (tirage aléatoire) (coût quasi équivalent à un passage de scan_batch complet, payé inutilement à chaque
//     tentative).
//   - Mode voisinage : parallélisme par TENTATIVE plutôt que par bloc à l'intérieur d'une
//     tentative (suppression de scan_segment/ThreadPool). L'ancien modèle répartissait les clés
//     d'UN SEUL segment entre les threads via des blocs ; pour un segment de voisinage typique
//     (quelques bits à quelques dizaines de bits, souvent < 2*HALF=1024 clés), il n'existait
//     qu'un seul bloc possible, donc un seul thread travaillait pendant que tous les autres
//     restaient inactifs (ex: -v3 avec 8 threads -> 7 threads inutilisés). Chaque thread
//     (neighbor_worker) traite désormais sa propre tentative indépendante de bout en bout, comme
//     le fait déjà le mode normal : tous les cœurs restent occupés quelle que soit la taille des
//     segments.
//
// Version 8 : le mode voisinage visite désormais les zones dans un ordre ALÉATOIRE SANS DOUBLON
//   (au lieu de tirages avec remise) : chaque « cellule » (motif des bits hors segment) est
//   scannée une seule fois, dans un ordre pseudo-aléatoire (permutation de Feistel à clés tirées à
//   chaque lancement). Le programme connaît donc exactement ce qui est déjà scanné : affichage
//   « essai X/Y | pourcentage | ETA » et message « Plage entièrement scannée » en fin de plage,
//   comme le mode normal. Plus aucun verrou ni générateur aléatoire pendant la recherche.
//
// Version 7 : suppression complète de la suite de Sobol (struct Sobol1D, option -sobol, verrou
//   g_sobol_mtx). Le mode voisinage tire désormais ses clés de référence de façon purement
//   probabiliste : tirages uniformes INDÉPENDANTS (avec remise) dans [0, span), via un générateur
//   mt19937_64 PROPRE À CHAQUE THREAD, graine issue de std::random_device. Plus aucun verrou
//   partagé pendant le tirage ; le mode normal (permutation aléatoire sans doublon) est inchangé.
//
// Version 6 : recherche de PLUSIEURS puzzles en un seul balayage (-pApB).
//   - Nouvel argument  -pApB  (ex : -p66p70) : cible TOUS les puzzles de A à B inclus. L'intervalle
//     parcouru est l'enveloppe commune : borne inférieure du plus petit puzzle (2^(A-1)) jusqu'à la
//     borne supérieure du plus grand (2^B - 1). Chaque clé générée n'est calculée qu'UNE fois et
//     comparée à toutes les cibles : chercher 5 puzzles coûte le même temps que chercher le plus grand.
//   - -pP seul (un puzzle) reste valable ; plusieurs arguments -p peuvent être combinés
//     (ex : -p66 -p70p72) : cibles = union des puzzles choisis, intervalle = du plus petit au plus grand.
//   - TargetSet : ensemble de cibles hash160 avec filtre binaire de 65536 bits (8 Ko, cache L1) :
//     coût par clé quasi identique à une cible unique, même avec 80 cibles.
//   - La recherche NE S'ARRÊTE PLUS à la première clé trouvée : elle continue jusqu'à ce que toutes
//     les cibles soient trouvées (ou plage épuisée / Ctrl-C). Chaque clé est enregistrée dans God.txt
//     avec son numéro de puzzle, et l'affichage indique « trouvées X/N ».
//   - scan_batch rappelle désormais un callback à CHAQUE correspondance (au lieu de s'arrêter à la
//     première), ce qui est indispensable en multi-cibles.
//   - Correctif : l'auto-test n'ajoute plus 8192 clés fictives au compteur du mode normal.
//
// Version 9 : moteur GPU (CUDA) pour le mode normal (non voisinage), multi-GPU.
//   - Compilation inchangée avec g++ (voir ligne de commande ci-dessus) : programme 100% CPU,
//     identique en tout point aux versions précédentes.
//   - Compilation avec nvcc (voir ligne ci-dessous) : ajoute un chemin GPU utilisé automatiquement
//     quand au moins un GPU CUDA est détecté (désactivable avec --cpu, forçable avec --gpu).
//     L'arithmétique de corps fini, les points de courbe, la table de base fixe, la technique de
//     fenêtre ±i (batch d'inversion) et le hash160 sont réimplémentés comme fonctions __device__,
//     en reproduisant très directement la logique déjà testée des fonctions CPU correspondantes
//     (mêmes formules, même structure), avec une fenêtre plus petite par thread (HALF_GPU, voir
//     plus bas) adaptée à la mémoire locale limitée d'un thread CUDA. Chaque chunk (au sens des
//     fonctions claim/worker déjà existantes) est traité par UN thread GPU indépendant ; les deux
//     GPU Kaggle T4 se partagent dynamiquement les chunks via claim_range() (verrou partagé,
//     identique en esprit à claim()). Chaque correspondance potentielle trouvée sur GPU est
//     revérifiée sur CPU par la fonction verify_key() déjà existante avant d'être enregistrée :
//     un bug de portage GPU peut donc au pire faire manquer une clé, jamais en inventer une fausse.
//     Un auto-test GPU dédié (gpu_run_self_test) tourne une fois par GPU avant toute recherche
//     réelle ; un GPU dont l'auto-test échoue est simplement ignoré (le CPU ou l'autre GPU
//     prennent le relais).
//
// Version 10 : le mode voisinage (-v/-vv/-vvv/-vvvv) utilise désormais lui aussi le(s) GPU par
//   défaut s'ils sont détectés et validés (gpu_neighbor_worker) : c'est EXACTEMENT le même moteur
//   de scan que le mode normal (GpuJob/GpuHit/gpu_scan_kernel, non modifiés), seule la table de
//   fenêtre change — multiples de 2^start_bit * G au lieu de multiples de G, puisqu'un segment de
//   voisinage de N bits avance par pas de 2^start_bit (voir build_gpu_win_host). L'auto-test GPU
//   vérifie désormais aussi ce cas (pas « grand », type segment) en plus du cas normal (pas = G).
//   Compilation CPU (inchangée)   : g++ -O3 -march=native -std=c++17 -o puzzle_pur puzzle_pur.cpp -pthread
//   Compilation GPU (Kaggle, 2x T4, Turing = sm_75) :
//     nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -O3,-march=native,-pthread -o puzzle_gpu puzzle_gpu.cu
//
// Version 11 : deux correctifs de performance GPU (vitesse mesurée très en dessous de l'attendu).
//   - GPU_HALF (fenêtre d'inversion par lot, device) remonté de 16 à 64 par défaut. À 16, le coût
//     de fe_inv (~270 multiplications équivalentes) n'était amorti que sur 16 clés (~17 mult/clé),
//     contre ~0.26 mult/clé côté CPU (fenêtre ±i, 2*HALF=1024 clés/inversion) : un surcoût ~60x par
//     clé qui, à lui seul, pouvait expliquer la majeure partie de l'écart. Reste surchargeable sans
//     toucher au fichier via -DGPU_HALF=N (voir commentaire détaillé au site de la macro) : à
//     valider empiriquement sur le GPU cible (clés/s + nvcc -Xptxas -v pour surveiller le spill
//     registre -> mémoire locale quand GPU_HALF grandit).
//   - gpu_worker/gpu_neighbor_worker : la préparation des jobs (un scalar_mul_fixed par job, soit
//     une inversion modulaire complète + jusqu'à 16 additions mixtes CHACUN) tournait en boucle
//     séquentielle sur l'unique thread hôte du GPU, celui-ci — et le GPU — restant par ailleurs
//     inactifs pendant ce calcul (aucun worker CPU n'est démarré en mode GPU, donc les autres
//     coeurs ne faisaient rien non plus). Parallélisée sur plusieurs threads CPU via le nouveau
//     helper prepare_jobs_parallel (réutilise le paramètre -t, sans effet jusqu'ici en mode GPU) ;
//     scalar_mul_fixed est une fonction pure (table g_gtable figée après démarrage), donc aucune
//     synchronisation n'est nécessaire au-delà du partitionnement en sous-plages disjointes.
//
// Version 12 : gpu_worker (mode normal) passe en DOUBLE TAMPON (2 jeux de buffers device + 2
//   streams CUDA). Avant ce changement, chaque lancement restait entièrement séquentiel : préparer
//   les jobs (CPU) -> copier -> lancer le kernel -> attendre sa fin (cudaDeviceSynchronize) -> lire
//   les résultats -> recommencer, le GPU restant inactif pendant toute la phase CPU même une fois
//   celle-ci parallélisée (version 11). Désormais, le kernel du lancement courant est soumis de
//   façon asynchrone (pas d'attente immédiate) : le CPU prépare et soumet le lancement SUIVANT
//   pendant que le précédent tourne encore sur le GPU, puis seulement à ce moment-là on attend la
//   fin du précédent (cudaStreamSynchronize) pour en lire les résultats. Tant que l'exécution GPU
//   d'un lancement dure au moins aussi longtemps que la préparation CPU du suivant (quasi toujours
//   vrai ici), ce temps CPU disparaît presque entièrement du temps total. GPU_HALF remonté à 128
//   (270/128 ≈ 2.1 mult/clé d'amortissement d'inversion, contre ~17 avant la version 11) : ce
//   second correctif rendait ce palier supplémentaire plus rentable en libérant le CPU du chemin
//   critique. Détail de comptabilité (claim_range/g_cur/safe_index) et limite sur la latence de
//   Ctrl-C : voir le commentaire détaillé au-dessus de gpu_worker.
//
// Version 13 : même double tampon porté à gpu_neighbor_worker (mode voisinage). La séquence des
//   lancements (lot de cellules x segment, ex. 3 lancements par lot en -vvvv) est traitée comme une
//   file linéaire de "rounds" : le CPU prépare le round suivant pendant que le précédent tourne
//   encore sur le GPU, exactement comme gpu_worker. Pas de claim_range/g_cur ici (ce chemin ne s'en
//   sert pas ; la progression affichée lit directement g_nb_next, déjà le cas avant ce changement),
//   mais g_nb_abandoned (condition du message final "plage entièrement scannée") est conservé
//   fidèle à la version à un seul tampon : il repasse à true dès qu'un lot de cellules réservé n'a
//   pas vu la totalité de ses segments traités (arrêt demandé en cours de lot, ou erreur CUDA
//   fatale — y compris en drainant proprement un round déjà soumis avant de l'abandonner).
//
// Version 14 : arithmétique de corps Fp côté device portée en limbes 32 bits (DU256 passe de
//   u64 d[4] à u32 v[8]), d'après puzzle.cu (fourni par l'utilisateur) : le GPU n'a pas de
//   multiplieur natif 64x64->128, contrairement au CPU ; cette opération devait de toute façon être
//   décomposée par nvcc en plusieurs multiplications 32x32->64 natives, mais de façon générique,
//   sans pouvoir enchaîner les retenues aussi étroitement qu'un calcul écrit DIRECTEMENT en limbes
//   32 bits. d_fe_add/d_fe_sub/d_fe_reduce utilisent en plus une correction modulo p PAR SÉLECTION
//   SANS BRANCHEMENT (masque tout-à-1/tout-à-0) au lieu d'un `if (geq(...)) sub_raw(...)`, pour
//   éviter la divergence de warp (threads d'un même warp exécutant en lock-step). d_fe_inv (chaîne
//   d'addition) et tous les appelants (gpu_scan_device, gpu_selftest_kernel, d_aff_add_full,
//   d_hash160, d_targets_find) sont inchangés dans leur STRUCTURE — seuls les accès directs aux
//   anciens d[0..3] ont été adaptés (v[2*i]=bas, v[2*i+1]=haut de l'ancien d[i] ; toD/fromD font
//   maintenant cette conversion au lieu d'une simple copie). d_adc/d_sbb/d_add_raw/d_sub_raw/d_mac
//   (ancienne version) supprimés, devenus inutilisés. Validation : n'ayant ni GPU ni nvcc ici, la
//   nouvelle arithmétique a été testée à part, sur CPU, dans un harnais indépendant comparant 2
//   millions de tirages aléatoires (add/sub/mul/sqr/inv, plus les cas limites 0 et 1) entre cette
//   version et la référence U256/fe_* 64 bits déjà validée par self_test() : aucun écart. Ceci ne
//   remplace pas une compilation nvcc réelle ni l'auto-test GPU intégré (gpu_run_self_test), à
//   lancer en conditions réelles avant toute recherche.
//
// Version 15 (v31) : inversion modulaire MUTUALISÉE À L'ÉCHELLE DU BLOC + préparation des jobs accélérée.
//   - Kernels : l'inversion par lot n'est plus faite par thread (1 d_fe_inv par thread et par fenêtre) mais
//     UNE FOIS par bloc via un arbre de produits en mémoire partagée (d_block_batch_inv : montée, 1 inversion
//     à la racine, descente). gpu_scan_kernel et gpu_selftest_kernel partagent le même code (gpu_block_scan)
//     et ne diffèrent que par le « puits » qui reçoit chaque clé. Conséquences : GPU_HALF par défaut 128 -> 32
//     (pre[] 4 Ko -> 1 Ko par thread), table de fenêtre en mémoire partagée, zero_flag[] -> masque de bits,
//     __launch_bounds__. Tous les threads d'un bloc exécutent le même nombre de barrières (nombre de fenêtres
//     uniforme par atomicMax, plus de return anticipé : un thread sans job contribue 1 à l'arbre).
//     La taille de bloc GPU_BT (puissance de 2) est fixée à la compilation : --gpu-threads est ignoré.
//   - Auto-test GPU : 2 blocs x GPU_BT threads (longueurs de job variées), comparaison de CHAQUE point et hash160
//     à la référence CPU, puis 2e étape avec le VRAI kernel de recherche (3 cibles à des positions délicates).
//   - Préparation des jobs (CPU) : mode normal = progression arithmétique des points de départ (une addition
//     par job + une inversion par sous-plage, build_progression_starts) ; mode voisinage = k*G Jacobien +
//     conversion affine par lot (jac_batch_to_aff). Avant : une multiplication scalaire complète + une inversion
//     PAR JOB (131 072 jobs par lancement avec la grille par défaut), probable goulot d'étranglement.
//   - Erreurs de lancement de kernel détectées (cudaGetLastError) au lieu d'être ignorées silencieusement.
//   - Validation sans GPU : g++ -DMELANIE_GPU_SIM -I. (cuda_sim.h : threads CUDA = fibres, mémoire partagée et
//     __syncthreads() réels, détection des barrières divergentes) ; voir run_sim_tests.sh.
//
// Version 16 (fichier .cu) : MODE HYBRIDE — TOUS les cœurs CPU ET TOUS les GPU travaillent EN MÊME TEMPS.
//   Compilation (adapter sm_XX : 75 = T4, 80 = A100, 86 = RTX 30xx, 89 = RTX 40xx, 90 = H100) :
//     nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -O3,-march=native,-pthread -o melanie Melanie_v32.cu
//     (multi-architecture : -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 ...)
//   - Mode normal (aléatoire sans doublon, --seq, -pP, -pApB, plage début:fin) : les threads CPU (worker) et les threads
//     pilotes de GPU (gpu_worker, un par GPU) puisent DYNAMIQUEMENT dans la MÊME file de chunks (claim/claim_range) :
//     chaque processeur prend ce qu'il peut traiter, l'équilibrage de charge est automatique (GPU rapides,
//     GPU lents, CPU), sans doublon ni oubli. Avant : dès qu'un GPU était détecté, les coeurs CPU ne scannaient plus.
//   - Mode voisinage (-v/-vv/-vvv/-vvvv) : même principe sur la file de cellules (g_nb_next) : neighbor_worker (CPU) et
//     gpu_neighbor_worker (GPU) avancent ensemble sur la même permutation de Feistel, sans doublon.
//   - Taille de grille GPU AUTOMATIQUE par carte (nombre de SM x blocs résidents x 8 vagues, via l'occupation réelle du
//     kernel) quand --gpu-blocks n'est pas donné : un mélange de GPU de puissances différentes est bien exploité.
//   - Attente de fin de kernel BLOQUANTE (cudaEventBlockingSync) au lieu d'un cudaStreamSynchronize en spin : le thread
//     pilote d'un GPU ne consomme plus 100 % d'un coeur CPU, qui reste donc disponible pour les workers CPU.
//   - Auto-tests GPU lancés EN PARALLÈLE (un thread par carte) : démarrage plus rapide avec plusieurs GPU.
//   - Si un GPU subit une erreur CUDA fatale en cours de route, la recherche continue sur le CPU et les autres GPU, mais
//     le message final ne prétend plus « plage entièrement scannée » (g_gpu_lost).
//   - Affichage : répartition instantanée « (GPU x/s, CPU y/s) » en mode hybride.
//   - Nouvelle option --gpu-only : GPU seuls (comportement des versions précédentes) ; --cpu : CPU seul ; sans option :
//     hybride si au moins un GPU valide est détecté, CPU seul sinon.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// ---- Sélection du moteur GPU ----
//   nvcc                     : vrai moteur CUDA
//   g++ -DMELANIE_GPU_SIM    : MÊME code compilé sur CPU avec le simulateur cuda_sim.h (threads CUDA = fibres,
//                              mémoire partagée et __syncthreads() réels) : sert à valider les kernels sans GPU
//   g++ (défaut)             : programme 100 % CPU
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define MEL_GPU 1
#define GPU_LAUNCH(k, g, b, s, ...) k<<<(g), (b), 0, (s)>>>(__VA_ARGS__)
#elif defined(MELANIE_GPU_SIM)
#include "cuda_sim.h"
#define MEL_GPU 1
#else
#define MEL_GPU 0
#endif

// Paramètres du moteur GPU (surchargeables : nvcc ... -DGPU_HALF=32 -DGPU_BT=128 -DGPU_MINBLK=3).
//   GPU_HALF   : nombre de clés consécutives par thread et par fenêtre (1..64). Grâce à l'inversion mutualisée
//                par bloc (voir gpu_block_scan), il n'a plus besoin d'être grand : 32 par défaut.
//   GPU_BT     : threads par bloc = largeur de l'arbre d'inversion (PUISSANCE DE 2, 32..512). Fixé à la
//                compilation (tableaux en mémoire partagée) : --gpu-threads est donc ignoré.
//   GPU_MINBLK : indication __launch_bounds__ (blocs résidents minimum par SM) pour borner les registres.
#ifndef GPU_HALF
#define GPU_HALF 32
#endif
#ifndef GPU_BT
#define GPU_BT 128
#endif
#ifndef GPU_MINBLK
#define GPU_MINBLK 3
#endif
static_assert(GPU_HALF >= 1 && GPU_HALF <= 64, "GPU_HALF doit etre dans [1,64]");
static_assert(GPU_BT >= 32 && GPU_BT <= 512 && (GPU_BT & (GPU_BT - 1)) == 0, "GPU_BT doit etre une puissance de 2 dans [32,512]");

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

// =================== U256 : entier 256 bits à 4 limbes FIXES ===================
struct U256 { u64 d[4]; };
static const U256 ZERO = {{0,0,0,0}};
static const U256 ONE  = {{1,0,0,0}};
static const u64  KC = 0x1000003D1ULL; // 2^256 mod p (p = 2^256 - 2^32 - 977)

static inline bool is_zero(const U256 &a) { return (a.d[0]|a.d[1]|a.d[2]|a.d[3]) == 0; }
static inline bool eq(const U256 &a, const U256 &b) {
    return ((a.d[0]^b.d[0])|(a.d[1]^b.d[1])|(a.d[2]^b.d[2])|(a.d[3]^b.d[3])) == 0;
}
static inline u64 add_raw(U256 &r, const U256 &a, const U256 &b) {
    u128 acc = 0; u64 c = 0;
    for (int i = 0; i < 4; i++) { acc = (u128)a.d[i] + b.d[i] + c; r.d[i] = (u64)acc; c = (u64)(acc >> 64); }
    return c;
}
static inline u64 sub_raw(U256 &r, const U256 &a, const U256 &b) {
    u64 br = 0;
    for (int i = 0; i < 4; i++) {
        u128 t = (u128)a.d[i] - b.d[i] - br;
        r.d[i] = (u64)t; br = (u64)((t >> 64) & 1);
    }
    return br;
}
static int highest_bit(const U256 &a) {
    for (int i = 255; i >= 0; --i) if ((a.d[i >> 6] >> (i & 63)) & 1) return i;
    return -1;
}
static inline int get_bit(const U256 &a, int i) { return (int)((a.d[i >> 6] >> (i & 63)) & 1); }

static bool parse_hex(const std::string &s, U256 &out) {
    if (s.empty() || s.size() > 64) return false;
    out = ZERO;
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[s.size() - 1 - i]; int v;
        if (c >= '0' && c <= '9') v = c - '0'; else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10; else return false;
        out.d[i / 16] |= (u64)v << ((i % 16) * 4);
    }
    return true;
}
static U256 u128_to_U256(u128 v) { U256 r; r.d[0] = (u64)v; r.d[1] = (u64)(v >> 64); r.d[2] = 0; r.d[3] = 0; return r; }

// =================== Corps fini Fp (secp256k1) ===================
static inline void fe_add(U256 &r, const U256 &a, const U256 &b) {
    U256 t; u64 c = add_raw(t, a, b);
    if (c) { U256 k = {{KC,0,0,0}}; add_raw(t, t, k); }
    // t >= p ? soustraire p — comparaison directe plutôt que de stocker p entier à part
    static const U256 P = {{0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL}};
    bool geq = true;
    for (int i = 3; i >= 0 && geq; --i) if (t.d[i] != P.d[i]) { geq = t.d[i] > P.d[i]; break; }
    if (geq) sub_raw(t, t, P);
    r = t;
}
static inline void fe_sub(U256 &r, const U256 &a, const U256 &b) {
    U256 t; u64 br = sub_raw(t, a, b);
    if (br) { U256 k = {{KC,0,0,0}}; sub_raw(t, t, k); }
    r = t;
}
static inline void fe_reduce(U256 &r, const u64 *t) {
    u64 r5[5]; u64 carry = 0;
    for (int i = 0; i < 4; i++) { u128 acc = (u128)t[4+i]*KC + t[i] + carry; r5[i] = (u64)acc; carry = (u64)(acc>>64); }
    r5[4] = carry;
    U256 res; u128 acc = (u128)r5[4]*KC + r5[0];
    res.d[0] = (u64)acc; carry = (u64)(acc>>64);
    for (int i = 1; i < 4; i++) { acc = (u128)r5[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    if (carry) {
        acc = (u128)res.d[0]+KC; res.d[0]=(u64)acc; carry=(u64)(acc>>64);
        for (int i = 1; i < 4; i++) { acc=(u128)res.d[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    }
    static const U256 P = {{0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL}};
    bool geq = true;
    for (int i = 3; i >= 0; --i) if (res.d[i] != P.d[i]) { geq = res.d[i] > P.d[i]; break; }
    if (geq) sub_raw(res, res, P);
    r = res;
}
static inline void fe_mul(U256 &r, const U256 &a, const U256 &b) {
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) { u128 acc=(u128)a.d[i]*b.d[j]+t[i+j]+carry; t[i+j]=(u64)acc; carry=(u64)(acc>>64); }
        t[i+4] = carry;
    }
    fe_reduce(r, t);
}
static inline void fe_sqr(U256 &r, const U256 &a) { // carré dédié : 10 produits au lieu de 16
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++) {
        u64 carry = 0;
        for (int j = i+1; j < 4; j++) { u128 acc=(u128)a.d[i]*a.d[j]+t[i+j]+carry; t[i+j]=(u64)acc; carry=(u64)(acc>>64); }
        t[i+4] = carry;
    }
    u64 c = 0;
    for (int k = 0; k < 8; k++) { u64 v=t[k]; t[k]=(v<<1)|c; c=v>>63; }
    u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        u128 acc=(u128)a.d[i]*a.d[i]+t[2*i]+carry; t[2*i]=(u64)acc; carry=(u64)(acc>>64);
        acc=(u128)t[2*i+1]+carry; t[2*i+1]=(u64)acc; carry=(u64)(acc>>64);
    }
    fe_reduce(r, t);
}
static inline void fe_sqr_n(U256 &r, const U256 &a, int n) { U256 t=a; for(int i=0;i<n;i++) fe_sqr(t,t); r=t; }
// Inversion par chaîne d'addition fixe (équivalent à a^(p-2) mod p, 255 carrés + 14 produits)
static U256 fe_inv(const U256 &a) {
    U256 x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(x2,a); fe_mul(x2,x2,a);
    fe_sqr(x3,x2); fe_mul(x3,x3,a);
    fe_sqr_n(x6,x3,3); fe_mul(x6,x6,x3);
    fe_sqr_n(x9,x6,3); fe_mul(x9,x9,x3);
    fe_sqr_n(x11,x9,2); fe_mul(x11,x11,x2);
    fe_sqr_n(x22,x11,11); fe_mul(x22,x22,x11);
    fe_sqr_n(x44,x22,22); fe_mul(x44,x44,x22);
    fe_sqr_n(x88,x44,44); fe_mul(x88,x88,x44);
    fe_sqr_n(x176,x88,88); fe_mul(x176,x176,x88);
    fe_sqr_n(x220,x176,44); fe_mul(x220,x220,x44);
    fe_sqr_n(x223,x220,3); fe_mul(x223,x223,x3);
    fe_sqr_n(t,x223,23); fe_mul(t,t,x22);
    fe_sqr_n(t,t,5); fe_mul(t,t,a);
    fe_sqr_n(t,t,3); fe_mul(t,t,x2);
    fe_sqr_n(t,t,2); fe_mul(t,t,a);
    return t;
}

// =================== Courbe secp256k1 (y^2 = x^3 + 7) ===================
struct Point { U256 x, y; bool inf = false; };
struct Jac { U256 X, Y, Z; bool inf = false; };

static Point G;

static Point aff_add(const Point &P, const Point &Q) {
    if (P.inf) return Q;
    if (Q.inf) return P;
    U256 l;
    if (eq(P.x, Q.x)) {
        if (!eq(P.y, Q.y) || is_zero(P.y)) return Point{ZERO, ZERO, true};
        U256 n, d, n3; fe_sqr(n, P.x); fe_add(n3, n, n); fe_add(n3, n3, n);
        fe_add(d, P.y, P.y); fe_mul(l, n3, fe_inv(d));
    } else {
        U256 n, d; fe_sub(n, Q.y, P.y); fe_sub(d, Q.x, P.x); fe_mul(l, n, fe_inv(d));
    }
    Point R; R.inf = false;
    fe_sqr(R.x, l); fe_sub(R.x, R.x, P.x); fe_sub(R.x, R.x, Q.x);
    U256 t; fe_sub(t, P.x, R.x); fe_mul(R.y, l, t); fe_sub(R.y, R.y, P.y);
    return R;
}
static Jac jac_dbl(const Jac &P) {
    if (P.inf || is_zero(P.Y)) return Jac{ZERO, ZERO, ZERO, true};
    U256 A,B,C,D,E,F,t,t2;
    fe_sqr(A,P.X); fe_sqr(B,P.Y); fe_sqr(C,B);
    fe_add(t,P.X,B); fe_sqr(t,t); fe_sub(t,t,A); fe_sub(t,t,C); fe_add(D,t,t);
    fe_add(E,A,A); fe_add(E,E,A); fe_sqr(F,E);
    Jac R; R.inf=false;
    fe_add(t,D,D); fe_sub(R.X,F,t);
    fe_sub(t,D,R.X); fe_mul(t,E,t);
    fe_add(t2,C,C); fe_add(t2,t2,t2); fe_add(t2,t2,t2);
    fe_sub(R.Y,t,t2);
    fe_mul(t,P.Y,P.Z); fe_add(R.Z,t,t);
    return R;
}
static Jac jac_madd(const Jac &P, const Point &Q) {
    if (P.inf) { Jac R; R.X=Q.x; R.Y=Q.y; R.Z=ONE; R.inf=false; return R; }
    U256 Z1Z1,U2,S2,H,HH,I,J,r,V,t;
    fe_sqr(Z1Z1,P.Z); fe_mul(U2,Q.x,Z1Z1);
    fe_mul(S2,Q.y,P.Z); fe_mul(S2,S2,Z1Z1);
    fe_sub(H,U2,P.X);
    fe_sub(t,S2,P.Y); fe_add(r,t,t);
    if (is_zero(H)) { if (is_zero(r)) return jac_dbl(P); return Jac{ZERO,ZERO,ZERO,true}; }
    fe_sqr(HH,H); fe_add(I,HH,HH); fe_add(I,I,I);
    fe_mul(J,H,I); fe_mul(V,P.X,I);
    Jac R; R.inf=false;
    fe_sqr(R.X,r); fe_sub(R.X,R.X,J); fe_sub(R.X,R.X,V); fe_sub(R.X,R.X,V);
    fe_sub(t,V,R.X); fe_mul(t,r,t);
    U256 u; fe_mul(u,P.Y,J); fe_add(u,u,u);
    fe_sub(R.Y,t,u);
    fe_add(t,P.Z,H); fe_sqr(t,t); fe_sub(t,t,Z1Z1); fe_sub(R.Z,t,HH);
    return R;
}
static Jac jac_mul(const U256 &k, const Point &P) {
    Jac R; R.X=R.Y=R.Z=ZERO; R.inf=true;
    for (int i = highest_bit(k); i >= 0; --i) { R = jac_dbl(R); if (get_bit(k,i)) R = jac_madd(R,P); }
    return R;
}
static Point jac_to_aff(const Jac &J) {
    Point A; A.x=ZERO; A.y=ZERO;
    if (J.inf) { A.inf=true; return A; }
    U256 zi=fe_inv(J.Z), zi2, zi3;
    fe_sqr(zi2,zi); fe_mul(zi3,zi2,zi);
    fe_mul(A.x,J.X,zi2); fe_mul(A.y,J.Y,zi3); A.inf=false;
    return A;
}

// =================== Table de multiplication à base fixe (calcul rapide de k*G) ===================
// 16 fenêtres de 8 bits ; chaque fenêtre contient les 255 multiples non nuls de 2^(8w)*G.
// Construite une seule fois au démarrage (4080 additions affines, ~20 ms) ; scalar_mul_fixed
// combine ensuite au plus 16 points par addition Jacobienne mixte (une seule inversion finale).
static const int GT_W = 8, GT_BITS = 128, GT_WINDOWS = GT_BITS / GT_W, GT_SIZE = 1 << GT_W;
static std::vector<Point> g_gtable;

static void init_gtable(const Point &g) {
    g_gtable.assign((size_t)GT_WINDOWS * GT_SIZE, Point{ZERO, ZERO, true});
    Point base = g;
    for (int w = 0; w < GT_WINDOWS; w++) {
        Point cur = base;
        g_gtable[(size_t)w*GT_SIZE + 1] = cur;
        for (int d = 2; d < GT_SIZE; d++) { cur = aff_add(cur, base); g_gtable[(size_t)w*GT_SIZE + d] = cur; }
        base = aff_add(cur, base); // 256 * base précédente
    }
}
static Jac scalar_mul_fixed_jac(const U256 &k) { // k*G en coordonnées de Jacobi (aucune inversion)
    Jac acc; acc.X=acc.Y=acc.Z=ZERO; acc.inf=true;
    for (int w = 0; w < GT_WINDOWS; w++) {
        int pos = w * GT_W, limb = pos / 64;
        if (limb >= 4) break;
        unsigned d = (unsigned)((k.d[limb] >> (pos % 64)) & (GT_SIZE - 1));
        if (d) acc = jac_madd(acc, g_gtable[(size_t)w*GT_SIZE + d]);
    }
    return acc;
}
static Point scalar_mul_fixed(const U256 &k) { return jac_to_aff(scalar_mul_fixed_jac(k)); }

// Conversion Jacobi -> affine de n points avec UNE SEULE inversion (astuce de Montgomery : produit
// préfixe, une inversion, répartition arrière) : ~4 multiplications par point au lieu d'une inversion
// complète (~270 mult. équivalentes). Les points à l'infini (J[i].inf) sont rendus tels quels (out.inf).
static void jac_batch_to_aff(const Jac *J, Point *out, size_t n, std::vector<U256> &pre) {
    if (n == 0) return;
    pre.resize(n);
    U256 acc = ONE;
    for (size_t i = 0; i < n; i++) { if (!J[i].inf) fe_mul(acc, acc, J[i].Z); pre[i] = acc; }
    U256 inv = fe_inv(acc);
    for (size_t i = n; i-- > 0;) {
        if (J[i].inf) { out[i] = Point{ZERO, ZERO, true}; continue; }
        U256 zi, zi2, zi3;
        if (i > 0) { fe_mul(zi, inv, pre[i-1]); fe_mul(inv, inv, J[i].Z); } else zi = inv;
        fe_sqr(zi2, zi); fe_mul(zi3, zi2, zi);
        Point A; fe_mul(A.x, J[i].X, zi2); fe_mul(A.y, J[i].Y, zi3); A.inf = false; out[i] = A;
    }
}

// Points de départ de 'count' chunks CONSÉCUTIFS (indices first_idx, first_idx+1, ...) du mode normal.
// L'offset du chunk idx est ((idx*stride) & (nchunks-1)) << cb : quand idx augmente de 1, il augmente de
// D = (stride mod nchunks) << cb, modulo M = nchunks << cb. Le point de départ suivant est donc le
// précédent + D*G (ou + (D-M)*G = -(M-D)*G en cas de repli modulo M) : UNE addition Jacobienne mixte par
// job au lieu d'une multiplication scalaire complète (~10 additions + 1 inversion), puis une seule
// inversion pour tout le lot. La préparation des jobs côté CPU était le goulot d'étranglement probable
// du GPU (131 072 jobs par lancement avec la grille par défaut) ; elle devient négligeable.
static void build_progression_starts(u128 lo, int cb, uint64_t stride, uint64_t nchunks,
                                     uint64_t first_idx, uint64_t count, Point *out) {
    if (count == 0) return;
    const uint64_t mask = nchunks - 1, ds = stride & mask;
    uint64_t o = (first_idx * stride) & mask;
    Jac cur = scalar_mul_fixed_jac(u128_to_U256(lo + ((u128)o << cb)));
    std::vector<Jac> J((size_t)count);
    Point Q1{ZERO, ZERO, true}, Q2{ZERO, ZERO, true};
    if (count > 1 && ds != 0) {
        const u128 D = (u128)ds << cb, M = (u128)nchunks << cb;
        Q1 = scalar_mul_fixed(u128_to_U256(D));
        Q2 = scalar_mul_fixed(u128_to_U256(M - D));
        fe_sub(Q2.y, ZERO, Q2.y); // Q2 = -(M-D)*G
    }
    for (uint64_t j = 0; j < count; j++) {
        J[(size_t)j] = cur;
        if (j + 1 < count && ds != 0) {
            o += ds;
            if (o >= nchunks) { o -= nchunks; cur = jac_madd(cur, Q2); } else cur = jac_madd(cur, Q1);
        }
    }
    std::vector<U256> scratch;
    jac_batch_to_aff(J.data(), out, (size_t)count, scratch);
}

// =================== SHA-256 (un seul bloc, messages <= 55 octets) ===================
static inline u32 rotr32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
static inline u32 rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
static const u32 SHA_K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static inline void sha256_block(const u32* w16, u32* out) {
    u32 w[64];
    for (int i=0;i<16;i++) w[i]=w16[i];
    for (int i=16;i<64;i++) {
        u32 s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    for (int i=0;i<64;i++) {
        u32 S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25), ch=(e&f)^(~e&g);
        u32 t1=h+S1+ch+SHA_K[i]+w[i];
        u32 S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22), mj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    out[0]=a+0x6a09e667; out[1]=b+0xbb67ae85; out[2]=c+0x3c6ef372; out[3]=d+0xa54ff53a;
    out[4]=e+0x510e527f; out[5]=f+0x9b05688c; out[6]=g+0x1f83d9ab; out[7]=h+0x5be0cd19;
}
static void sha256_bytes(const unsigned char* m, size_t n, unsigned char* out32) {
    unsigned char buf[64]; memset(buf,0,64); memcpy(buf,m,n); buf[n]=0x80;
    u64 bl=(u64)n*8; for (int i=0;i<8;i++) buf[63-i]=(unsigned char)(bl>>(8*i));
    u32 w[16],o[8];
    for (int i=0;i<16;i++) w[i]=((u32)buf[4*i]<<24)|((u32)buf[4*i+1]<<16)|((u32)buf[4*i+2]<<8)|buf[4*i+3];
    sha256_block(w,o);
    for (int i=0;i<8;i++){ out32[4*i]=o[i]>>24; out32[4*i+1]=o[i]>>16; out32[4*i+2]=o[i]>>8; out32[4*i+3]=o[i]; }
}
static void sha256d(const unsigned char* d, size_t n, unsigned char out[32]) {
    unsigned char t[32]; sha256_bytes(d,n,t); sha256_bytes(t,32,out);
}

// =================== RIPEMD-160 (un seul bloc, scalaire) ===================
static const unsigned char RMD_R1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
static const unsigned char RMD_R2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
static const unsigned char RMD_S1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
static const unsigned char RMD_S2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
static const u32 RMD_K1[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
static const u32 RMD_K2[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
template<int F> static inline u32 rmd_f(u32 x,u32 y,u32 z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
static inline void rmd160_block(const u32* X, u32* out) {
    u32 a=0x67452301,b=0xEFCDAB89,c=0x98BADCFE,d=0x10325476,e=0xC3D2E1F0;
    u32 a2=a,b2=b,c2=c,d2=d,e2=e;
    auto seg=[&](int F){
        for (int j=16*F;j<16*F+16;j++) {
            u32 t = rotl32(a + (F==0?rmd_f<0>(b,c,d):F==1?rmd_f<1>(b,c,d):F==2?rmd_f<2>(b,c,d):F==3?rmd_f<3>(b,c,d):rmd_f<4>(b,c,d)) + X[RMD_R1[j]] + RMD_K1[F], RMD_S1[j]) + e;
            a=e; e=d; d=rotl32(c,10); c=b; b=t;
            int G2 = 4-F;
            u32 t2 = rotl32(a2 + (G2==0?rmd_f<0>(b2,c2,d2):G2==1?rmd_f<1>(b2,c2,d2):G2==2?rmd_f<2>(b2,c2,d2):G2==3?rmd_f<3>(b2,c2,d2):rmd_f<4>(b2,c2,d2)) + X[RMD_R2[j]] + RMD_K2[F], RMD_S2[j]) + e2;
            a2=e2; e2=d2; d2=rotl32(c2,10); c2=b2; b2=t2;
        }
    };
    seg(0); seg(1); seg(2); seg(3); seg(4);
    u32 t=0xEFCDAB89u+c+d2;
    out[1]=0x98BADCFEu+d+e2; out[2]=0x10325476u+e+a2; out[3]=0xC3D2E1F0u+a+b2; out[4]=0x67452301u+b+c2; out[0]=t;
}
static void hash160_bytes(const unsigned char* m, size_t n, unsigned char* out20) {
    unsigned char dg[32]; sha256_bytes(m,n,dg);
    u32 X[16]={0},o[5];
    for (int i=0;i<8;i++) X[i]=(u32)dg[4*i]|((u32)dg[4*i+1]<<8)|((u32)dg[4*i+2]<<16)|((u32)dg[4*i+3]<<24);
    X[8]=0x80; X[14]=256;
    rmd160_block(X,o);
    for (int i=0;i<5;i++){ out20[4*i]=o[i]; out20[4*i+1]=o[i]>>8; out20[4*i+2]=o[i]>>16; out20[4*i+3]=o[i]>>24; }
}
static U256 pack20(const unsigned char* b) {
    U256 r=ZERO; memcpy(&r.d[0],b,8); memcpy(&r.d[1],b+8,8); u32 t; memcpy(&t,b+16,4); r.d[2]=t; return r;
}

#if MEL_GPU
// =================== Moteur GPU (CUDA) : arithmétique et hachage __device__ ===================
// Réimplémentation séparée (plutôt que des qualificateurs __host__ __device__ sur les fonctions
// CPU ci-dessus) : formules RECOPIÉES À L'IDENTIQUE des fonctions CPU correspondantes (mêmes
// opérations, même ordre), afin de pouvoir se relire en comparant ligne à ligne. Un auto-test
// dédié (gpu_run_self_test, plus bas) exécute ce code sur un lot de points connus et compare
// CHAQUE coordonnée et CHAQUE hash160 obtenu à la valeur calculée par le code CPU déjà validé ;
// un GPU dont l'auto-test échoue est exclu de la recherche (voir main()). Chaque correspondance
// potentielle remontée par le GPU est de toute façon revérifiée par la fonction CPU verify_key()
// avant tout enregistrement (voir gpu_worker) : un bug de portage GPU peut donc au pire faire
// manquer une clé, jamais en enregistrer une fausse.
//
// ARITHMÉTIQUE EN LIMBES 32 BITS (portée de puzzle.cu, fourni par l'utilisateur) — remplace la
// version précédente en limbes 64 bits (avec __umul64hi, voir l'ancien commentaire ci-dessus,
// conservé au-dessus pour le contexte général sur l'absence de u128 côté device). Le GPU n'a pas de
// multiplieur natif 64x64->128 ; cette multiplication devait de toute façon être décomposée par le
// compilateur en plusieurs multiplications 32x32->64 natives — mais de façon générique, sans
// pouvoir enchaîner les retenues aussi étroitement qu'un calcul écrit DIRECTEMENT en limbes 32
// bits (ce que fait ce portage). v[0] = poids faible ; l'ancien d[i] (64 bits) correspond à
// v[2*i] (bas, 32 bits) | v[2*i+1] (haut, 32 bits) — voir toD/fromD plus loin pour la conversion.
//
// Deux différences par rapport à l'ancienne version, au-delà de la largeur des limbes :
//  - d_fe_add/d_fe_sub/d_fe_reduce font la correction modulo p par SÉLECTION SANS BRANCHEMENT
//    (masque tout-à-1 ou tout-à-0 appliqué par ET/OU bit à bit), au lieu d'un `if (geq(...))
//    sub_raw(...)`. Sur GPU, où les threads d'un même warp exécutent en lock-step, un `if` dont la
//    condition diffère d'un thread à l'autre (ce qui arrive forcément ici selon les valeurs
//    traitées par chaque thread) sérialise les deux branches pour TOUT le warp (divergence) ; la
//    sélection par masque ne coûte rien de plus qu'un ET/OU, exécuté identiquement par tous les
//    threads quelle que soit l'issue individuelle.
//  - d_fe_sqr fait 36 produits 32 bits au lieu de 64 (même principe que fe_sqr côté CPU : exploiter
//    a[i]*a[j]==a[j]*a[i] pour ne calculer qu'une fois chaque produit croisé, doublé ensuite).
// Justesse : mêmes formules, même ordre d'opérations que puzzle.cu, qui les teste bit à bit contre
// sa propre référence 64 bits dans son auto-test interne. Ici, l'auto-test GPU dédié déjà existant
// (gpu_run_self_test, plus bas, inchangé) joue exactement ce rôle : il compare chaque coordonnée et
// hash160 obtenus par ce nouveau code à la valeur CPU déjà validée (fe_mul/fe_sqr/fe_inv, 64 bits,
// jamais touchés) ; un bug de portage ferait échouer ce test avant toute recherche réelle, et de
// toute façon chaque correspondance potentielle est de toute façon revérifiée côté CPU
// (verify_key()) avant tout enregistrement.
struct DU256 { u32 v[8]; };
struct DPoint { DU256 x, y; };

__device__ __forceinline__ DU256 d_zero() { DU256 r; 
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}
__device__ __forceinline__ DU256 d_one() { DU256 r = d_zero(); r.v[0] = 1; return r; }
__device__ __forceinline__ bool d_is_zero(const DU256 &a) {
    u32 o = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) o |= a.v[i];
    return o == 0;
}
__device__ __forceinline__ bool d_eq(const DU256 &a, const DU256 &b) {
    u32 x = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) x |= a.v[i] ^ b.v[i];
    return x == 0;
}
// a+b mod p (p = 2^256-2^32-977, donc 2^256 ≡ 2^32+977 (mod p)) : addition brute 8 limbes, puis
// correction par sélection sans branchement. La constante 2^32+977 en limbes 32 bits : poids 0 ->
// 977 (0x3D1), poids 1 -> 1 (le "2^32"), le reste 0.
__device__ __forceinline__ void d_fe_add(DU256 &r, const DU256 &a, const DU256 &b) {
    u32 t[8], u[8]; u64 c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += (u64)a.v[i] + b.v[i]; t[i] = (u32)c; c >>= 32; }
    const u32 c1 = (u32)c;
    c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += (u64)t[i] + (i==0 ? 0x3D1u : (i==1 ? 1u : 0u)); u[i] = (u32)c; c >>= 32; }
    const u32 sel = 0u - (c1 | (u32)c); // t a débordé de 2^256, OU t+KC déborde (t était déjà >= p) -> on garde u
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = (u[i] & sel) | (t[i] & ~sel);
}
// a-b mod p : soustraction brute, puis si emprunt on retranche encore 2^32+977 (ce qui, après
// l'emprunt sur 2^256, revient à ajouter p) ; seul le SIGNE de l'emprunt (0 ou tout-à-1) sert de
// masque, jamais sa valeur dans un `if`.
__device__ __forceinline__ void d_fe_sub(DU256 &r, const DU256 &a, const DU256 &b) {
    u32 t[8]; u64 br = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { u64 d = (u64)a.v[i] - b.v[i] - br; t[i] = (u32)d; br = (d>>63)&1; }
    const u32 m = 0u - (u32)br;
    u64 bo = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u64 k = (i==0) ? (u64)(0x3D1u & m) : ((i==1) ? (u64)(1u & m) : 0ull);
        u64 d = (u64)t[i] - k - bo; r.v[i] = (u32)d; bo = (d>>63)&1;
    }
}
// Réduction d'un produit de 512 bits (16 limbes 32 bits) modulo p : repliement 2^256=2^32+977
// appliqué deux fois (le premier repliement peut encore légèrement déborder de 2^256, d'où le
// second, portant alors sur une valeur minuscule), puis mise en forme canonique (< p) par la même
// sélection sans branchement que d_fe_add.
__device__ __forceinline__ void d_fe_reduce(DU256 &r, const u32 *t) {
    u32 s[8], u[8]; u64 c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u64 cur = (u64)t[8+i]*977u + t[i] + c;
        if (i > 0) cur += t[7+i];
        s[i] = (u32)cur; c = cur >> 32;
    }
    const u64 top = c + (u64)t[15];
    u64 acc = (u64)s[0] + top*977u; s[0] = (u32)acc; acc >>= 32;
    acc += (u64)s[1] + top; s[1] = (u32)acc; acc >>= 32;
    #pragma unroll
    for (int i = 2; i < 8; i++) { acc += s[i]; s[i] = (u32)acc; acc >>= 32; }
    {
        const u32 m = 0u - (u32)acc; // retenue du repliement précédent : on la replie à son tour (valeur alors minuscule)
        acc = (u64)s[0] + (0x3D1u & m); s[0] = (u32)acc; acc >>= 32;
        acc += (u64)s[1] + (1u & m); s[1] = (u32)acc; acc >>= 32;
        #pragma unroll
        for (int i = 2; i < 8; i++) { acc += s[i]; s[i] = (u32)acc; acc >>= 32; }
    }
    acc = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc += (u64)s[i] + (i==0 ? 0x3D1u : (i==1 ? 1u : 0u)); u[i] = (u32)acc; acc >>= 32; }
    const u32 sel = 0u - (u32)acc; // s >= p <=> s + (2^32+977) déborde de 2^256 <=> s - p = u
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = (u[i] & sel) | (s[i] & ~sel);
}
__device__ __forceinline__ void d_fe_mul(DU256 &r, const DU256 &a, const DU256 &b) {
    u32 t[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) t[i] = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u64 carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) { u64 cur = (u64)a.v[i]*b.v[j] + t[i+j] + carry; t[i+j] = (u32)cur; carry = cur >> 32; }
        t[i+8] = (u32)carry;
    }
    d_fe_reduce(r, t);
}
// Carré dédié : 36 produits 32 bits au lieu de 64 (même principe que fe_sqr côté CPU, transposé).
__device__ __forceinline__ void d_fe_sqr(DU256 &r, const DU256 &a) {
    u32 t[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) t[i] = 0;
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        u64 carry = 0;
        #pragma unroll
        for (int j = i+1; j < 8; j++) { u64 cur = (u64)a.v[i]*a.v[j] + t[i+j] + carry; t[i+j] = (u32)cur; carry = cur >> 32; }
        t[i+8] = (u32)carry;
    }
    u32 c = 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) { u32 v = t[k]; t[k] = (v<<1)|c; c = v>>31; }
    u64 carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u64 sq = (u64)a.v[i]*a.v[i];
        u64 cur = (u64)t[2*i] + (u32)sq + carry; t[2*i] = (u32)cur; carry = cur >> 32;
        cur = (u64)t[2*i+1] + (u32)(sq>>32) + carry; t[2*i+1] = (u32)cur; carry = cur >> 32;
    }
    d_fe_reduce(r, t);
}
__device__ __forceinline__ void d_fe_sqr_n(DU256 &r, const DU256 &a, int n) {
    DU256 t = a; for (int i = 0; i < n; i++) d_fe_sqr(t, t); r = t;
}
// Chaîne d'addition identique à fe_inv (CPU) : mêmes exposants, même ordre d'opérations.
__device__ DU256 d_fe_inv(const DU256 &a) {
    DU256 x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    d_fe_sqr(x2,a); d_fe_mul(x2,x2,a);
    d_fe_sqr(x3,x2); d_fe_mul(x3,x3,a);
    d_fe_sqr_n(x6,x3,3); d_fe_mul(x6,x6,x3);
    d_fe_sqr_n(x9,x6,3); d_fe_mul(x9,x9,x3);
    d_fe_sqr_n(x11,x9,2); d_fe_mul(x11,x11,x2);
    d_fe_sqr_n(x22,x11,11); d_fe_mul(x22,x22,x11);
    d_fe_sqr_n(x44,x22,22); d_fe_mul(x44,x44,x22);
    d_fe_sqr_n(x88,x44,44); d_fe_mul(x88,x88,x44);
    d_fe_sqr_n(x176,x88,88); d_fe_mul(x176,x176,x88);
    d_fe_sqr_n(x220,x176,44); d_fe_mul(x220,x220,x44);
    d_fe_sqr_n(x223,x220,3); d_fe_mul(x223,x223,x3);
    d_fe_sqr_n(t,x223,23); d_fe_mul(t,t,x22);
    d_fe_sqr_n(t,t,5); d_fe_mul(t,t,a);
    d_fe_sqr_n(t,t,3); d_fe_mul(t,t,x2);
    d_fe_sqr_n(t,t,2); d_fe_mul(t,t,a);
    return t;
}
// Addition affine pleine (P != Q et P != -Q supposé non dégénéré ; cas dégénérés gérés par
// l'appelant pour le chemin rapide de gpu_scan_device, comme en CPU dans aff_add()).
__device__ __forceinline__ DPoint d_aff_add_full(const DPoint &P, const DPoint &Q) {
    DU256 n, d, l; d_fe_sub(n, Q.y, P.y); d_fe_sub(d, Q.x, P.x); d_fe_mul(l, n, d_fe_inv(d));
    DPoint R; d_fe_sqr(R.x, l); d_fe_sub(R.x, R.x, P.x); d_fe_sub(R.x, R.x, Q.x);
    DU256 t; d_fe_sub(t, P.x, R.x); d_fe_mul(R.y, l, t); d_fe_sub(R.y, R.y, P.y);
    return R;
}

// ---- SHA-256 + RIPEMD-160 (un seul bloc, scalaire) : copie directe de sha256_block/rmd160_block ----
__constant__ u32 D_SHA_K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__constant__ unsigned char D_RMD_R1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
__constant__ unsigned char D_RMD_R2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
__constant__ unsigned char D_RMD_S1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
__constant__ unsigned char D_RMD_S2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
__constant__ u32 D_RMD_K1[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
__constant__ u32 D_RMD_K2[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
__device__ __forceinline__ u32 d_rotr32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
__device__ __forceinline__ u32 d_rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
template<int F> __device__ __forceinline__ u32 d_rmd_f(u32 x,u32 y,u32 z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
__device__ __forceinline__ void d_rmd160_block(const u32* X, u32* out) {
    u32 a=0x67452301,b=0xEFCDAB89,c=0x98BADCFE,d=0x10325476,e=0xC3D2E1F0;
    u32 a2=a,b2=b,c2=c,d2=d,e2=e;
    #pragma unroll
    for (int F = 0; F < 5; F++) {
        int G2 = 4-F;
        #pragma unroll
        for (int jj = 0; jj < 16; jj++) {
            int j = 16*F+jj;
            u32 fv = (F==0?d_rmd_f<0>(b,c,d):F==1?d_rmd_f<1>(b,c,d):F==2?d_rmd_f<2>(b,c,d):F==3?d_rmd_f<3>(b,c,d):d_rmd_f<4>(b,c,d));
            u32 t = d_rotl32(a + fv + X[D_RMD_R1[j]] + D_RMD_K1[F], D_RMD_S1[j]) + e;
            a=e; e=d; d=d_rotl32(c,10); c=b; b=t;
            u32 fv2 = (G2==0?d_rmd_f<0>(b2,c2,d2):G2==1?d_rmd_f<1>(b2,c2,d2):G2==2?d_rmd_f<2>(b2,c2,d2):G2==3?d_rmd_f<3>(b2,c2,d2):d_rmd_f<4>(b2,c2,d2));
            u32 t2 = d_rotl32(a2 + fv2 + X[D_RMD_R2[j]] + D_RMD_K2[F], D_RMD_S2[j]) + e2;
            a2=e2; e2=d2; d2=d_rotl32(c2,10); c2=b2; b2=t2;
        }
    }
    u32 t=0xEFCDAB89u+c+d2;
    out[1]=0x98BADCFEu+d+e2; out[2]=0x10325476u+e+a2; out[3]=0xC3D2E1F0u+a+b2; out[4]=0x67452301u+b+c2; out[0]=t;
}
// hash160 d'une clé publique compressée (33 octets : préfixe de parité 02/03 + X sur 32 octets)
// construite directement à partir des limbes de X et de la parité, sans buffer d'octets
// intermédiaire (même principe que hash160_vec côté CPU, mais pour UNE SEULE clé scalaire).
__device__ void d_hash160(const DU256 &x, u64 parity, DU256 &out) {
    u32 Xw[8];
    // x.v[7] = mot de 32 bits le plus significatif ... x.v[0] = le moins significatif : c'est
    // directement l'ordre big-endian attendu par SHA-256, sans conversion (contre un découpage
    // explicite 64->32 bits dans l'ancienne représentation).
    #pragma unroll
    for (int i = 0; i < 8; i++) Xw[i] = x.v[7-i];
    u32 P = 0x02u + (u32)parity;
    u32 w[64];
    w[0] = (P<<24) | (Xw[0]>>8);
    #pragma unroll
    for (int i = 1; i < 8; i++) w[i] = (Xw[i-1]<<24) | (Xw[i]>>8);
    w[8] = (Xw[7]<<24) | 0x00800000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 264;
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        u32 s0 = d_rotr32(w[i-15],7)^d_rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1 = d_rotr32(w[i-2],17)^d_rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,dd=0xa54ff53au,e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 S1=d_rotr32(e,6)^d_rotr32(e,11)^d_rotr32(e,25), ch=(e&f)^(~e&g);
        u32 t1=h+S1+ch+D_SHA_K[i]+w[i];
        u32 S0=d_rotr32(a,2)^d_rotr32(a,13)^d_rotr32(a,22), mj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+mj;
        h=g; g=f; f=e; e=dd+t1; dd=c; c=b; b=a; a=t1+t2;
    }
    u32 H[8];
    H[0]=a+0x6a09e667u; H[1]=b+0xbb67ae85u; H[2]=c+0x3c6ef372u; H[3]=dd+0xa54ff53au;
    H[4]=e+0x510e527fu; H[5]=f+0x9b05688cu; H[6]=g+0x1f83d9abu; H[7]=h+0x5be0cd19u;
    u32 rw[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 q=H[i];
        rw[i]=(q>>24)|((q>>8)&0x0000FF00u)|((q<<8)&0x00FF0000u)|(q<<24);
    }
    rw[8]=0x80; for(int i=9;i<14;i++) rw[i]=0; rw[14]=256; rw[15]=0;
    u32 o[5]; d_rmd160_block(rw, o);
    out.v[0]=o[0]; out.v[1]=o[1]; out.v[2]=o[2]; out.v[3]=o[3]; out.v[4]=o[4];
    out.v[5]=0; out.v[6]=0; out.v[7]=0;
}

// ---- Table de cibles côté device : copie de TargetSet::find() (filtre + comparaison complète) ----
__device__ __forceinline__ int d_targets_find(const DU256 &x, const DU256* th, int n_targets, const u64* filter) {
    u64 ix = x.v[0] & 0xFFFFu;
    if (!((filter[ix>>6] >> (ix&63)) & 1ull)) return -1;
    for (int i = 0; i < n_targets; ++i) if (d_eq(th[i], x)) return i;
    return -1;
}

// ---- Moteur de scan device : fenêtre à sens unique + INVERSION MODULAIRE MUTUALISÉE PAR BLOC ----
//
// Chaque thread parcourt des clés consécutives par fenêtres de GPU_HALF clés (point C, puis C+1*step ...
// C+GPU_HALF*step ; table 'win' = multiples 1..GPU_HALF de 'step'). Le passage de C à C+i*step est une addition
// affine, qui exige 1/(win[i].x - C.x) : on regroupe ces GPU_HALF inversions par l'astuce de Montgomery.
//
//   AVANT : chaque thread inversait SON produit de GPU_HALF différences : un d_fe_inv (~270 multiplications
//           équivalentes) par thread et par fenêtre, soit ~2 mult./clé à GPU_HALF=128, et des tableaux pre[]
//           de 4 Ko par thread (trafic de mémoire locale important).
//   MAINTENANT : chaque thread calcule toujours son produit préfixe pre[] et son produit total 'acc', mais
//           l'inversion des GPU_BT produits d'un bloc est faite UNE SEULE FOIS par un arbre de produits en
//           mémoire partagée (d_block_batch_inv) :
//             montée   : noeud k = produit(noeud 2k, noeud 2k+1)          (log2(GPU_BT) paliers parallèles)
//             racine   : le thread 0 fait l'unique d_fe_inv du bloc
//             descente : inv(2k) = inv(k)*prod(2k+1) ; inv(2k+1) = inv(k)*prod(2k)   (idem, parallèle)
//           soit 1 inversion + ~3*GPU_BT multiplications par bloc et par fenêtre, au lieu de GPU_BT inversions :
//           le coût d'inversion par clé tombe à ~270/(GPU_BT*GPU_HALF) + quelques mult. d'arbre, négligeable
//           même avec GPU_HALF=32 (pre[] : 1 Ko/thread). La table de fenêtre est recopiée en mémoire partagée
//           (lecture en diffusion : tous les threads lisent la même entrée au même moment).
//
// RÈGLES DE CORRECTION (le simulateur cuda_sim.h les vérifie) :
//   - __syncthreads() doit être exécuté par TOUS les threads du bloc, le même nombre de fois : aucun `return`
//     anticipé avant la fin (un thread sans job participe quand même, avec un produit neutre = 1) ;
//   - le nombre de fenêtres est donc UNIFORME dans le bloc (max des nombres de fenêtres de ses threads,
//     obtenu par atomicMax en mémoire partagée) ; un thread dont le job est terminé contribue 1 et n'émet rien.
// Un seul code (gpu_block_scan) sert le kernel de recherche ET le kernel d'auto-test : ce que l'auto-test valide
// est exactement ce qui tourne en production ; seul le « puits » (sink) qui reçoit chaque clé diffère.

struct GpuHit { int job; u64 offset; int target; };

// Un job : un bloc de 'n' clés consécutives (pas = 'step' de la table de fenêtre) à partir du point affine
// 'start' (calculé par le CPU). AUCUNE clé privée n'est manipulée ici, seulement des points et des décalages
// relatifs au job (reconvertis en clé privée réelle côté CPU, à partir du point de départ connu).
struct GpuJob { DPoint start; u64 n; };

// Arbre de produits : limbes en SoA (s[limbe][noeud]) pour que les threads consécutifs touchent des banques
// de mémoire partagée distinctes. Noeuds 1..2*GPU_BT-1 ; feuilles = GPU_BT..2*GPU_BT-1 (feuille GPU_BT+t = thread t).
typedef u32 ShNode[8][2 * GPU_BT];
__device__ __forceinline__ DU256 sh_get(const ShNode &s, int i) {
    DU256 r;
    #pragma unroll
    for (int k = 0; k < 8; k++) r.v[k] = s[k][i];
    return r;
}
__device__ __forceinline__ void sh_put(ShNode &s, int i, const DU256 &x) {
    #pragma unroll
    for (int k = 0; k < 8; k++) s[k][i] = x.v[k];
}

// Retourne 1/acc pour le thread courant, en une seule inversion pour tout le bloc. À appeler par TOUS les
// threads du bloc. Chaque thread ne modifie que ses propres noeuds par palier ; une barrière sépare les paliers.
// (Nul ne relit une feuille d'une fenêtre précédente : la dernière barrière de la descente précède toute lecture,
// et chaque thread n'écrit sa feuille suivante qu'après l'avoir lue.)
__device__ __forceinline__ DU256 d_block_batch_inv(ShNode &node, const DU256 &acc) {
    const unsigned t = threadIdx.x;
    sh_put(node, GPU_BT + (int)t, acc);
    __syncthreads();
    #pragma unroll
    for (int span = GPU_BT / 2; span >= 1; span >>= 1) {
        if (t < (unsigned)span) {
            const int k = span + (int)t;
            DU256 p; d_fe_mul(p, sh_get(node, 2*k), sh_get(node, 2*k + 1));
            sh_put(node, k, p);
        }
        __syncthreads();
    }
    if (t == 0) sh_put(node, 1, d_fe_inv(sh_get(node, 1))); // l'UNIQUE inversion du bloc (produit total, jamais nul)
    __syncthreads();
    #pragma unroll
    for (int span = 1; span < GPU_BT; span <<= 1) {
        if (t < (unsigned)span) {
            const int k = span + (int)t;
            const DU256 iv = sh_get(node, k), l = sh_get(node, 2*k), r = sh_get(node, 2*k + 1);
            DU256 il, ir;
            d_fe_mul(il, iv, r); // 1/l = 1/(l*r) * r
            d_fe_mul(ir, iv, l); // 1/r = 1/(l*r) * l
            sh_put(node, 2*k, il); sh_put(node, 2*k + 1, ir);
        }
        __syncthreads();
    }
    return sh_get(node, GPU_BT + (int)t);
}

// Puits « recherche » : hash160 de chaque clé, comparaison aux cibles (filtre + comparaison complète), remontée
// des correspondances (au plus max_hits par lancement ; jamais atteint en pratique).
struct GpuHitSink {
    const DU256* th; int n_targets; const u64* filter; int job; GpuHit* hits; int* hit_count; int max_hits;
    __device__ __forceinline__ void operator()(u64 offset, const DU256 &x, u64 par) {
        DU256 h; d_hash160(x, par, h);
        int ti = d_targets_find(h, th, n_targets, filter);
        if (ti >= 0) {
            int idx = atomicAdd(hit_count, 1);
            if (idx < max_hits) { hits[idx].job = job; hits[idx].offset = offset; hits[idx].target = ti; }
        }
    }
};
// Puits « auto-test » : enregistre CHAQUE clé (x, parité) et son hash160 pour comparaison avec la référence CPU.
struct GpuRecSink {
    DU256* out_x; u64* out_par; DU256* out_hash; u64 base;
    __device__ __forceinline__ void operator()(u64 offset, const DU256 &x, u64 par) {
        out_x[base + offset] = x; out_par[base + offset] = par; d_hash160(x, par, out_hash[base + offset]);
    }
};

// Scanne le job du thread courant (n clés à partir de 'start') en coopération avec le bloc. Chaque clé est
// transmise à sink(offset_relatif_au_job, x, parité_de_y).
template <class Sink>
__device__ __forceinline__ void gpu_block_scan(const DPoint* __restrict__ win, const DPoint &start, u64 n, Sink &sink) {
    __shared__ ShNode sh_node;
    __shared__ DPoint sh_win[GPU_HALF];
    __shared__ unsigned long long sh_nwin;

    const unsigned t = threadIdx.x;
    for (unsigned i = t; i < (unsigned)GPU_HALF; i += GPU_BT) sh_win[i] = win[i];
    if (t == 0) sh_nwin = 0ull;
    __syncthreads();
    atomicMax(&sh_nwin, (unsigned long long)((n + (u64)GPU_HALF - 1) / (u64)GPU_HALF));
    __syncthreads();
    const u64 nwin = sh_nwin; // nombre de fenêtres du bloc : identique pour tous les threads

    DU256 Cx = start.x, Cy = start.y;
    const DU256 ONEc = d_one();
    u64 done = 0;
    for (u64 wi = 0; wi < nwin; ++wi) {
        const bool active = done < n;
        const u64 rem = active ? (n - done) : 0ull;
        const u64 w = rem < (u64)GPU_HALF ? rem : (u64)GPU_HALF;       // 0 si le job de ce thread est terminé
        const bool is_last = active && (done + w == n);
        const int m = active ? (is_last ? (int)w - 1 : (int)w) : 0;     // additions nécessaires dans cette fenêtre

        // Décalage 0 = point C lui-même : coût nul, toujours une sortie valide.
        if (active) sink(done, Cx, (u64)(Cy.v[0] & 1u));

        // Produits préfixes des différences ; un delta nul (cas dégénéré, probabilité ~2^-256 pour une clé
        // aléatoire) est remplacé par 1 et mémorisé dans zmask : cette sortie est simplement sautée.
        DU256 pre[GPU_HALF + 1];
        u64 zmask = 0;
        DU256 acc = ONEc;
        for (int i = 1; i <= m; i++) {
            DU256 d; d_fe_sub(d, sh_win[i-1].x, Cx);
            if (d_is_zero(d)) { zmask |= 1ull << (i - 1); d = ONEc; }
            d_fe_mul(acc, acc, d); pre[i] = acc;
        }

        DU256 inv = d_block_batch_inv(sh_node, acc);  // 1/acc, une seule inversion pour tout le bloc

        DU256 nextCx = Cx, nextCy = Cy; // inchangés si m==0 (jamais utilisés dans ce cas)
        for (int i = m; i >= 1; --i) {
            const bool zf = (zmask >> (i - 1)) & 1ull;
            DU256 invd;
            if (i > 1) {
                d_fe_mul(invd, inv, pre[i-1]);
                DU256 d; d_fe_sub(d, sh_win[i-1].x, Cx); if (zf) d = ONEc;
                d_fe_mul(inv, inv, d);
            } else invd = inv;
            const bool is_jump = (i == m) && !is_last;
            if (zf) {
                if (is_jump) { nextCx = Cx; nextCy = Cy; }
                continue;
            }
            DU256 tt, l, x3, y3;
            d_fe_sub(tt, sh_win[i-1].y, Cy); d_fe_mul(l, tt, invd);
            d_fe_sqr(x3, l); d_fe_sub(x3, x3, Cx); d_fe_sub(x3, x3, sh_win[i-1].x);
            d_fe_sub(tt, Cx, x3); d_fe_mul(y3, l, tt); d_fe_sub(y3, y3, Cy);
            if (is_jump) { nextCx = x3; nextCy = y3; }
            else sink(done + (u64)i, x3, (u64)(y3.v[0] & 1u));
        }
        if (active && !is_last) { Cx = nextCx; Cy = nextCy; }
        done += w;
    }
}

__global__ void __launch_bounds__(GPU_BT, GPU_MINBLK)
gpu_scan_kernel(const DPoint* win, const GpuJob* jobs, int n_jobs,
                const DU256* target_h, int n_targets, const u64* filter,
                GpuHit* hits, int* hit_count, int max_hits) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const bool has = i < n_jobs;                 // PAS de return anticipé : voir règles de correction ci-dessus
    DPoint st; st.x = d_zero(); st.y = d_zero();
    u64 n = 0;
    if (has) { st = jobs[i].start; n = jobs[i].n; }
    GpuHitSink sink{target_h, n_targets, filter, i, hits, hit_count, max_hits};
    gpu_block_scan(win, st, n, sink);
}

// Auto-test GPU : CHAQUE thread de CHAQUE bloc parcourt son propre job (points de départ distincts, longueurs
// n variées : 0, 1, GPU_HALF-1, GPU_HALF, GPU_HALF+1, plusieurs fenêtres...) et enregistre chaque point
// (x, parité) et hash160 dans une zone de 'stride' entrées qui lui est propre. Plusieurs threads et plusieurs
// blocs sont indispensables : l'algorithme dépend de la coopération entre threads d'un même bloc.
__global__ void __launch_bounds__(GPU_BT, GPU_MINBLK)
gpu_selftest_kernel(const DPoint* win, const GpuJob* jobs, u64 stride,
                    DU256* out_x, u64* out_par, DU256* out_hash) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    GpuRecSink sink{out_x, out_par, out_hash, (u64)i * stride};
    gpu_block_scan(win, jobs[i].start, jobs[i].n, sink);
}
#endif // MEL_GPU

// ---- hash160 vectorisé (8 clés à la fois) : le cœur de l'accélération de ce moteur ----
#ifndef VLANES
#define VLANES 8
#endif
typedef u32 vu __attribute__((vector_size(4*VLANES)));
static inline vu vsplat(u32 x){ vu v; for(int i=0;i<VLANES;i++) v[i]=x; return v; }
#define VROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define VROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
template<int F> static inline vu rmd_fv(vu x,vu y,vu z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
template<int F> static inline void rmd_segv(const vu* X,vu& a,vu& b,vu& c,vu& d,vu& e,vu& a2,vu& b2,vu& c2,vu& d2,vu& e2){
    for (int j=16*F;j<16*F+16;j++) {
        vu t=VROTL(a+rmd_fv<F>(b,c,d)+X[RMD_R1[j]]+vsplat(RMD_K1[F]),RMD_S1[j])+e;
        a=e; e=d; d=VROTL(c,10); c=b; b=t;
        t=VROTL(a2+rmd_fv<4-F>(b2,c2,d2)+X[RMD_R2[j]]+vsplat(RMD_K2[F]),RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=VROTL(c2,10); c2=b2; b2=t;
    }
}
// hash160 de VLANES clés publiques compressées (xs[l], pars[l]) -> out[l] (empaqueté comme pack20)
static void hash160_vec(const U256* xs, const u64* pars, U256* out) {
    u32 Xl[8][VLANES], Pl[VLANES];
    for (int l=0;l<VLANES;l++) {
        const U256& x = xs[l];
        Xl[0][l]=(u32)(x.d[3]>>32); Xl[1][l]=(u32)x.d[3]; Xl[2][l]=(u32)(x.d[2]>>32); Xl[3][l]=(u32)x.d[2];
        Xl[4][l]=(u32)(x.d[1]>>32); Xl[5][l]=(u32)x.d[1]; Xl[6][l]=(u32)(x.d[0]>>32); Xl[7][l]=(u32)x.d[0];
        Pl[l]=0x02u+(u32)pars[l];
    }
    vu X[8],w[64],H[8],rw[16];
    for (int i=0;i<8;i++) memcpy(&X[i],Xl[i],sizeof(vu));
    vu P; memcpy(&P,Pl,sizeof(vu));
    w[0]=(P<<24)|(X[0]>>8);
    for (int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|vsplat(0x00800000u);
    for (int i=9;i<15;i++) w[i]=vsplat(0);
    w[15]=vsplat(264);
    for (int i=16;i<64;i++) {
        vu s0=VROTR(w[i-15],7)^VROTR(w[i-15],18)^(w[i-15]>>3);
        vu s1=VROTR(w[i-2],17)^VROTR(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    vu a=vsplat(0x6a09e667u),b=vsplat(0xbb67ae85u),c=vsplat(0x3c6ef372u),d=vsplat(0xa54ff53au);
    vu e=vsplat(0x510e527fu),f=vsplat(0x9b05688cu),g=vsplat(0x1f83d9abu),h=vsplat(0x5be0cd19u);
    for (int i=0;i<64;i++) {
        vu S1=VROTR(e,6)^VROTR(e,11)^VROTR(e,25), ch=(e&f)^(~e&g);
        vu t1=h+S1+ch+vsplat(SHA_K[i])+w[i];
        vu S0=VROTR(a,2)^VROTR(a,13)^VROTR(a,22), mj=(a&b)^(a&c)^(b&c);
        vu t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]=a+vsplat(0x6a09e667u); H[1]=b+vsplat(0xbb67ae85u); H[2]=c+vsplat(0x3c6ef372u); H[3]=d+vsplat(0xa54ff53au);
    H[4]=e+vsplat(0x510e527fu); H[5]=f+vsplat(0x9b05688cu); H[6]=g+vsplat(0x1f83d9abu); H[7]=h+vsplat(0x5be0cd19u);
    for (int i=0;i<8;i++) {
        vu q=H[i];
        rw[i]=(q>>24)|((q>>8)&vsplat(0x0000FF00u))|((q<<8)&vsplat(0x00FF0000u))|(q<<24);
    }
    rw[8]=vsplat(0x80); for(int i=9;i<14;i++) rw[i]=vsplat(0); rw[14]=vsplat(256); rw[15]=vsplat(0);
    vu a1=vsplat(0x67452301u),b1=vsplat(0xEFCDAB89u),c1=vsplat(0x98BADCFEu),d1=vsplat(0x10325476u),e1=vsplat(0xC3D2E1F0u);
    vu a2=a1,b2=b1,c2=c1,d2=d1,e2=e1;
    rmd_segv<0>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<1>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<2>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<3>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<4>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    vu o[5];
    o[0]=vsplat(0xEFCDAB89u)+c1+d2; o[1]=vsplat(0x98BADCFEu)+d1+e2; o[2]=vsplat(0x10325476u)+e1+a2;
    o[3]=vsplat(0xC3D2E1F0u)+a1+b2; o[4]=vsplat(0x67452301u)+b1+c2;
    u32 T[5][VLANES];
    for (int k=0;k<5;k++) memcpy(T[k],&o[k],sizeof(vu));
    for (int l=0;l<VLANES;l++) {
        out[l].d[0]=T[0][l]|((u64)T[1][l]<<32); out[l].d[1]=T[2][l]|((u64)T[3][l]<<32);
        out[l].d[2]=T[4][l]; out[l].d[3]=0;
    }
}

// =================== Base58Check (adresse P2PKH) et WIF ===================
static const char *B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static bool decode_address(const std::string &addr, unsigned char out[20]) {
    unsigned char buf[25] = {0};
    for (char c : addr) {
        const char *p = c ? strchr(B58, c) : nullptr;
        if (!p) return false;
        int carry = p - B58;
        for (int i = 24; i >= 0; --i) { carry += 58 * buf[i]; buf[i] = carry & 0xFF; carry >>= 8; }
        if (carry) return false;
    }
    unsigned char h[32]; sha256d(buf, 21, h);
    if (memcmp(h, buf + 21, 4) != 0 || buf[0] != 0x00) return false;
    memcpy(out, buf + 1, 20);
    return true;
}
static std::string base58_encode(const std::vector<unsigned char> &in) {
    std::vector<unsigned char> d(in.size() * 138 / 100 + 1, 0);
    size_t len = 0;
    for (unsigned char b : in) {
        int carry = b; size_t i = 0;
        for (auto it = d.rbegin(); (carry || i < len) && it != d.rend(); ++it, ++i) {
            carry += 256 * (*it); *it = carry % 58; carry /= 58;
        }
        len = i;
    }
    auto it = d.begin(); while (it != d.end() && *it == 0) ++it;
    std::string s(std::count(in.begin(), in.begin() + std::min<size_t>(1, in.size()), 0), '1');
    for (; it != d.end(); ++it) s += B58[*it];
    return s;
}
static void to_bytes32_priv(u128 k, unsigned char out[32]) {
    memset(out, 0, 32);
    for (int i = 31; i >= 16; --i) { out[i] = (unsigned char)(k & 0xFF); k >>= 8; }
}
static std::string to_hex(u128 k) {
    char t[40]; snprintf(t, sizeof t, "%016llx%016llx", (unsigned long long)(k>>64), (unsigned long long)k);
    std::string s(t); size_t p = s.find_first_not_of('0');
    return "0x" + (p == std::string::npos ? "0" : s.substr(p));
}
// Nombre de bits nécessaires pour représenter v (0 pour v == 0, 1 pour v == 1, etc.).
static int u128_bitlen(u128 v) {
    int n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}
static uint64_t next_pow2_u64(uint64_t x) {
    uint64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}
// Parse une valeur hexadécimale (préfixe "0x"/"0X" optionnel) dans un u128. Retourne false si la
// chaîne est vide, contient un caractère non hexadécimal, ou dépasse 128 bits (32 chiffres hex).
static bool parse_u128_hex(const std::string &in, u128 &out) {
    std::string s = in;
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    if (s.empty() || s.size() > 32) return false;
    u128 v = 0;
    for (char c : s) {
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (u128)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (u128)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (u128)(c - 'A' + 10);
        else return false;
    }
    out = v;
    return true;
}
// Parse un argument de plage "début:fin" en hexadécimal. Retourne false si le format est invalide.
static bool parse_hex_range(const std::string &arg, u128 &lo, u128 &hi) {
    size_t p = arg.find(':');
    if (p == std::string::npos) return false;
    u128 a, b;
    if (!parse_u128_hex(arg.substr(0, p), a) || !parse_u128_hex(arg.substr(p + 1), b)) return false;
    lo = std::min(a, b); hi = std::max(a, b);
    return true;
}
static std::string to_wif(u128 k) {
    std::vector<unsigned char> v(38);
    v[0] = 0x80; to_bytes32_priv(k, &v[1]); v[33] = 0x01;
    unsigned char h[32]; sha256d(v.data(), 34, h); memcpy(&v[34], h, 4);
    return base58_encode(v);
}

// =================== Moteur de scan ===================
static std::atomic<bool> g_stop{false}, g_found{false};
static std::atomic<uint64_t> g_count{0};
static std::atomic<uint64_t> g_count_gpu{0}; // part de g_count scannée par les GPU (affichage hybride)
static std::atomic<bool> g_gpu_lost{false};   // un GPU a subi une erreur fatale : des chunks réservés ont pu être perdus
static std::mutex g_mtx, g_file_mtx, g_print_mtx; // g_print_mtx protège stdout (progression vs "CLE TROUVEE")

// Ensemble de cibles (hash160). Un puzzle seul = 1 cible ; -pApB = jusqu'à 80 cibles cherchées
// simultanément pendant le MÊME balayage de l'intervalle commun. Un filtre binaire de 65536 bits
// (8 Ko, reste dans le cache L1) indexé par les 16 bits de poids faible du hash160 écarte d'emblée
// ~99,9 % des clés (80 cibles -> 0,12 % de faux positifs) avant toute comparaison complète.
static const int MAX_TARGETS = 80;
struct TargetSet {
    std::vector<U256> h;           // hash160 empaquetés (pack20)
    std::vector<std::string> addr; // adresse P2PKH de chaque cible
    std::vector<int> puzzle;       // numéro de puzzle (0 = adresse libre hors puzzle)
    u64 filter[1024] = {};
    void add(const std::string &a, const unsigned char h20[20], int pz) {
        h.push_back(pack20(h20)); addr.push_back(a); puzzle.push_back(pz);
        u64 ix = h.back().d[0] & 0xFFFF; filter[ix >> 6] |= 1ULL << (ix & 63);
    }
    size_t size() const { return h.size(); }
    // Indice de la cible égale à x, ou -1.
    inline int find(const U256 &x) const {
        u64 ix = x.d[0] & 0xFFFF;
        if (!((filter[ix >> 6] >> (ix & 63)) & 1)) return -1;
        for (size_t i = 0; i < h.size(); ++i) if (eq(h[i], x)) return (int)i;
        return -1;
    }
};
static TargetSet g_ts;
static std::atomic<bool> g_tfound[MAX_TARGETS]; // cible déjà trouvée (évite les doublons d'enregistrement)
static std::atomic<int>  g_remaining{0};        // nombre de cibles restant à trouver

static uint64_t g_next = 0, g_nchunks = 0;
static std::vector<uint64_t> g_cur;
static const uint64_t IDLE = UINT64_MAX;

static bool claim(unsigned t, uint64_t &idx) {
    std::lock_guard<std::mutex> l(g_mtx);
    if (g_next >= g_nchunks) return false;
    idx = g_next++; g_cur[t] = idx; return true;
}
static void release(unsigned t) { std::lock_guard<std::mutex> l(g_mtx); g_cur[t] = IDLE; }
static uint64_t safe_index() {
    std::lock_guard<std::mutex> l(g_mtx);
    uint64_t m = g_next; for (auto c : g_cur) m = std::min(m, c); return m;
}
// Variante de claim() pour le moteur GPU : réserve une PLAGE d'indices de chunks (un par thread
// CUDA du prochain lancement de kernel) au lieu d'un seul. Même sémantique que claim()/release()
// pour safe_index() : g_cur[t] = début de la plage en cours (plage non encore terminée).
static bool claim_range(unsigned t, uint64_t want, uint64_t &start_idx, uint64_t &count) {
    std::lock_guard<std::mutex> l(g_mtx);
    if (g_next >= g_nchunks) return false;
    start_idx = g_next;
    count = std::min<uint64_t>(want, g_nchunks - g_next);
    g_next += count;
    g_cur[t] = start_idx;
    return true;
}

// Sérialise la coordonnée X en 32 octets big-endian (SEC1).
static inline void x_to_bytes(const U256 &x, unsigned char out[32]) {
    for (int i = 0; i < 4; i++) {
        u64 limb = x.d[3 - i];
        for (int j = 0; j < 8; j++) out[i*8 + j] = (unsigned char)(limb >> (56 - 8*j));
    }
}
// Sérialise un point affine en clé publique compressée SEC1 (33 octets : préfixe de parité + X).
static inline void point_to_compressed(const Point &p, unsigned char out[33]) {
    out[0] = (p.y.d[0] & 1) ? 0x03 : 0x02;
    x_to_bytes(p.x, out + 1);
}
static bool verify_key(u128 k, int ti) {
    Point p = jac_to_aff(jac_mul(u128_to_U256(k), G));
    unsigned char comp[33], h[20];
    point_to_compressed(p, comp);
    hash160_bytes(comp, 33, h);
    return eq(pack20(h), g_ts.h[ti]);
}
static void save_key(u128 k, int ti) {
    const std::string &addr = g_ts.addr[ti];
    int pz = g_ts.puzzle[ti];
    std::string hex_k = to_hex(k);
    std::string wif_k = to_wif(k);
    std::lock_guard<std::mutex> l(g_file_mtx);
    std::ofstream f("God.txt", std::ios::app);
    f << "Adresse : " << addr << "\n";
    if (pz > 0) f << "Puzzle : #" << pz << "\n";
    f << "Clé privée (hex) : " << hex_k << "\n"
      << "WIF (compressé)  : " << wif_k << "\n"
      << "----------------------------------------\n";
    f.flush();
    {
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\n\n*** CLE TROUVEE ***\n");
        if (pz > 0) printf("Puzzle           : #%d\n", pz);
        printf("Adresse          : %s\n", addr.c_str());
        printf("Cle privee (hex) : %s\n", hex_k.c_str());
        printf("WIF (compresse)  : %s\n", wif_k.c_str());
        printf("Enregistree dans God.txt\n");
        fflush(stdout);
    }
}
// Traite une correspondance signalée par scan_batch : vérifie la clé en recalculant tout depuis
// zéro, l'enregistre UNE seule fois, et ne déclenche l'arrêt général que quand TOUTES les cibles
// sont trouvées (g_found == « toutes trouvées »).
static void handle_hit(u128 k, int ti) {
    if (!verify_key(k, ti)) return;
    if (g_tfound[ti].exchange(true)) return; // déjà trouvée (ex: tirage aléatoire qui repasse dessus)
    save_key(k, ti);
    if (g_remaining.fetch_sub(1) == 1) { g_found = true; g_stop = true; }
}
// Gestionnaire de signal (Ctrl-C / SIGTERM) : ne fait que des opérations async-signal-safe
// (std::atomic lock-free + write()), jamais de printf/cout/mutex, qui ne sont pas garantis
// sûrs à appeler depuis un gestionnaire de signal et pourraient bloquer (deadlock) s'il est
// déclenché pendant que le thread principal tient déjà le verrou d'affichage.
// 1er Ctrl-C  : demande un arrêt propre (g_stop) — les threads finissent leur segment en cours
//               (quelques centaines de µs à quelques ms au plus, vérifié très fréquemment) puis
//               sont rejoints normalement, et le résumé final est affiché.
// 2e Ctrl-C   : au cas où l'arrêt propre ne répondrait pas (ex: calcul bloqué), sortie immédiate.
static std::atomic<int> g_signal_count{0};
static void on_signal(int) {
    g_stop = true;
    if (g_signal_count.fetch_add(1) == 0) {
        static const char msg[] =
            "\n[Ctrl-C] Arret demande : fin du segment en cours puis sortie propre "
            "(Ctrl-C a nouveau pour forcer l'arret immediat)...\n";
        ssize_t r = write(STDOUT_FILENO, msg, sizeof(msg) - 1); (void)r;
    } else {
        static const char msg2[] = "\n[Ctrl-C] Arret force.\n";
        ssize_t r = write(STDOUT_FILENO, msg2, sizeof(msg2) - 1); (void)r;
        _exit(130);
    }
}
static double g_display_interval = 30.0; // secondes entre deux rafraîchissements de l'affichage (-s T)

// Formate un nombre de clés (ou de clés/s) avec l'unité adaptée : k, M, G, T, P.
static std::string fmt_keys(double v) {
    static const char *U[] = {"", "k", "M", "G", "T", "P"};
    int i = 0;
    double a = v;
    while (a >= 1000.0 && i < 5) { a /= 1000.0; ++i; }
    char b[32];
    if (i == 0) snprintf(b, sizeof b, "%.0f", a);
    else snprintf(b, sizeof b, "%.2f %s", a, U[i]);
    return b;
}
static std::string fmt_time(double s) {
    if (s > 3.15e9) return ">100 ans";
    char b[64];
    if (s > 3.15e7) snprintf(b, sizeof b, "%.1f ans", s / 3.15e7);
    else if (s > 86400) snprintf(b, sizeof b, "%.1f j", s / 86400);
    else snprintf(b, sizeof b, "%02d:%02d:%02d", (int)s/3600, ((int)s/60)%60, (int)s%60);
    return b;
}

// Fragment d'affichage « | trouvées X/N » (vide quand il n'y a qu'une seule cible).
static std::string fmt_targets() {
    int n = (int)g_ts.size();
    if (n <= 1) return "";
    return " | trouvées " + std::to_string(n - g_remaining.load()) + "/" + std::to_string(n);
}

// =================== Fenêtre ±i centrée : le cœur de la vitesse ===================
// Table des multiples 1..HALF du point de pas 'base' (G pour le mode par défaut, un point
// quelconque pour le mode voisinage), plus le pas vers le lot suivant (2*HALF*base).
static const int HALF = 512;
struct WinTable { std::vector<U256> gx, gy; U256 sx, sy; };

static void build_wintable(WinTable &wt, const Point &base) {
    wt.gx.assign(HALF + 1, ZERO); wt.gy.assign(HALF + 1, ZERO);

    // Accumule 1*base .. HALF*base, plus le point de pas suivant 2*HALF*base, en coordonnées de
    // Jacobi (jac_madd / jac_dbl : aucune inversion par point), puis convertit tout le lot en
    // affine avec UNE SEULE inversion (même astuce de lot que scan_batch), au lieu des HALF
    // inversions individuelles que faisait l'ancienne version via aff_add. C'est ce qui rendait
    // build_wintable très coûteuse en mode voisinage, où elle est reconstruite à chaque segment.
    const int M = HALF + 1; // +1 pour le point de pas suivant
    // Buffers réutilisés d'un appel à l'autre : build_wintable est rappelée à chaque segment en
    // mode voisinage, donc réallouer 4 vecteurs à chaque appel serait un coût fixe inutile.
    // Pas de risque de concurrence : build_wintable n'est jamais appelée depuis plusieurs threads
    // à la fois (une fois au démarrage, sinon séquentiellement depuis la boucle d'essais).
    static std::vector<U256> X(M + 1), Y(M + 1), Z(M + 1), pre(M + 1); // indices 1..M

    Jac cur; cur.X = base.x; cur.Y = base.y; cur.Z = ONE; cur.inf = false; // 1*base
    X[1] = cur.X; Y[1] = cur.Y; Z[1] = cur.Z;
    for (int i = 2; i <= HALF; i++) {
        cur = jac_madd(cur, base);
        X[i] = cur.X; Y[i] = cur.Y; Z[i] = cur.Z;
    }
    Jac s = jac_dbl(cur); // cur == HALF*base ici, donc s == 2*HALF*base
    X[M] = s.X; Y[M] = s.Y; Z[M] = s.Z;

    // Inversion de lot : produit préfixe, une seule fe_inv, puis répartition arrière.
    U256 acc = ONE;
    for (int i = 1; i <= M; i++) { fe_mul(acc, acc, Z[i]); pre[i] = acc; }
    U256 inv = fe_inv(acc);
    for (int i = M; i >= 1; --i) {
        U256 zi; // 1/Z[i]
        if (i > 1) { fe_mul(zi, inv, pre[i-1]); fe_mul(inv, inv, Z[i]); }
        else zi = inv;
        U256 zi2, zi3, ax, ay;
        fe_sqr(zi2, zi); fe_mul(zi3, zi2, zi);
        fe_mul(ax, X[i], zi2); fe_mul(ay, Y[i], zi3);
        if (i <= HALF) { wt.gx[i] = ax; wt.gy[i] = ay; } else { wt.sx = ax; wt.sy = ay; }
    }
}

struct Scratch {
    std::vector<U256> pre, cx; std::vector<char> zero; std::vector<u64> cpar; std::vector<int> coff;
    Scratch() : pre(HALF+2), cx(2*HALF+2*VLANES+2), zero(HALF+2), cpar(2*HALF+2*VLANES+2), coff(2*HALF+2*VLANES+2) {}
};

// Scanne N pas consécutifs à partir du point d'indice 0 'start', le long de la table 'wt'.
// Appelle on_hit(offset, indice_cible) pour CHAQUE clé dont le hash160 correspond à l'une des
// cibles de 'ts' (offset = position de la clé par rapport à 'start'), sans interrompre le balayage :
// indispensable quand plusieurs cibles sont cherchées à la fois. Retourne -1 si le balayage est
// terminé, -2 s'il a été interrompu par should_stop.
template <typename StopFn, typename HitFn>
static int64_t scan_batch(const WinTable &wt, const Point &start, uint64_t N, Scratch &sc,
                           std::atomic<uint64_t> &counter, const TargetSet &ts,
                           StopFn should_stop, HitFn on_hit) {
    Point C = aff_add(start, Point{wt.gx[HALF], wt.gy[HALF], false}); // point d'indice HALF
    uint64_t done = 0;
    while (done < N) {
        if (should_stop()) return -2;
        uint64_t count = std::min<uint64_t>((uint64_t)(2*HALF), N - done);
        bool last = count < (uint64_t)(2*HALF);
        int nc = 0;
        auto emit = [&](const U256 &x, u64 par, int off) { sc.cx[nc]=x; sc.cpar[nc]=par; sc.coff[nc]=off; nc++; };
        emit(C.x, C.y.d[0] & 1, 0);

        U256 acc = ONE;
        for (int i = 1; i <= HALF; i++) {
            U256 d; fe_sub(d, wt.gx[i], C.x);
            if (is_zero(d)) { sc.zero[i] = 1; d = ONE; } else sc.zero[i] = 0;
            fe_mul(acc, acc, d); sc.pre[i] = acc;
        }
        U256 dS; fe_sub(dS, wt.sx, C.x);
        bool zs = is_zero(dS); if (zs) dS = ONE;
        fe_mul(acc, acc, dS);
        U256 inv = fe_inv(acc);

        Point Cn{ZERO, ZERO, false};
        {
            U256 invd; fe_mul(invd, inv, sc.pre[HALF]); fe_mul(inv, inv, dS);
            if (!zs && !last) {
                U256 t, l; fe_sub(t, wt.sy, C.y); fe_mul(l, t, invd);
                fe_sqr(Cn.x, l); fe_sub(Cn.x, Cn.x, C.x); fe_sub(Cn.x, Cn.x, wt.sx);
                fe_sub(t, C.x, Cn.x); fe_mul(Cn.y, l, t); fe_sub(Cn.y, Cn.y, C.y);
            }
        }
        for (int i = HALF; i >= 1; --i) {
            U256 invd;
            if (i > 1) {
                fe_mul(invd, inv, sc.pre[i-1]);
                U256 d; fe_sub(d, wt.gx[i], C.x); if (sc.zero[i]) d = ONE;
                fe_mul(inv, inv, d);
            } else invd = inv;
            if (sc.zero[i]) { // cas dégénéré (rare) : repli sur l'addition affine directe
                U256 ny; fe_sub(ny, ZERO, wt.gy[i]);
                Point pm = aff_add(C, Point{wt.gx[i], ny, false});
                emit(pm.x, pm.y.d[0] & 1, -i);
                if (i < HALF) { Point pp = aff_add(C, Point{wt.gx[i], wt.gy[i], false}); emit(pp.x, pp.y.d[0] & 1, i); }
                continue;
            }
            U256 t, l, x3, y3;
            fe_add(t, wt.gy[i], C.y); fe_mul(l, t, invd);
            fe_sqr(x3, l); fe_sub(x3, x3, C.x); fe_sub(x3, x3, wt.gx[i]);
            fe_sub(t, x3, C.x); fe_mul(y3, l, t); fe_sub(y3, y3, C.y);
            emit(x3, y3.d[0] & 1, -i);
            if (i < HALF) {
                fe_sub(t, wt.gy[i], C.y); fe_mul(l, t, invd);
                fe_sqr(x3, l); fe_sub(x3, x3, C.x); fe_sub(x3, x3, wt.gx[i]);
                fe_sub(t, C.x, x3); fe_mul(y3, l, t); fe_sub(y3, y3, C.y);
                emit(x3, y3.d[0] & 1, i);
            }
        }

        for (int c = 0; c < nc; c += VLANES) {
            U256 xs[VLANES], hs[VLANES]; u64 ps[VLANES]; int m = std::min((int)VLANES, nc - c);
            for (int l = 0; l < VLANES; l++) { int idx = c + (l < m ? l : m - 1); xs[l] = sc.cx[idx]; ps[l] = sc.cpar[idx]; }
            hash160_vec(xs, ps, hs);
            for (int l = 0; l < m; l++) {
                int ti = ts.find(hs[l]);
                if (ti >= 0) on_hit((int64_t)done + HALF + sc.coff[c+l], ti);
            }
        }
        counter.fetch_add(count, std::memory_order_relaxed);
        done += count;
        if (last) break;
        if (zs) C = aff_add(C, Point{wt.sx, wt.sy, false}); else C = Cn;
    }
    return -1;
}

static WinTable g_wt_main; // table construite à partir de G, pour le mode par défaut (pas = 1 clé)

// lo/span délimitent la plage réelle à couvrir (span = fin - début + 1, pas nécessairement une
// puissance de deux pour une plage personnalisée "début:fin" en hexadécimal). g_nchunks reste une
// puissance de deux (nécessaire pour que le mélange par 'stride' soit une bijection sans doublon) :
// certains indices permutés tombent donc au-delà de span (reliquat de l'arrondi à la puissance de
// deux supérieure) et sont simplement sautés ; le dernier chunk réel, lui, est tronqué à la taille
// qu'il lui reste dans la plage.
static void worker(unsigned t, u128 lo, u128 span, int cb, uint64_t stride) {
    const uint64_t n_full = 1ULL << cb, mask = g_nchunks - 1;
    uint64_t idx;
    Scratch sc;
    while (!g_stop.load(std::memory_order_relaxed) && claim(t, idx)) {
        u128 offset = (u128)((idx * stride) & mask) << cb;
        if (offset >= span) { release(t); continue; } // chunk hors plage réelle : rien à scanner
        uint64_t n = (uint64_t)std::min<u128>((u128)n_full, span - offset);
        u128 start = lo + offset;
        Point start_pt = scalar_mul_fixed(u128_to_U256(start));
        bool finished = true;
        scan_batch(g_wt_main, start_pt, n, sc, g_count, g_ts, [&]() {
            if (g_stop.load(std::memory_order_relaxed)) { finished = false; return true; }
            return false;
        }, [&](int64_t hit, int ti) { handle_hit(start + (u128)hit, ti); });
        if (finished) release(t); else return;
    }
}

// =================== Moteur GPU (CUDA) : orchestration côté hôte ===================
// Tout le code qui manipule un type GPU (DPoint, DU256, ...) reste confiné à l'intérieur de
// #ifdef __CUDACC__ ; les fonctions ci-dessous (gpu_device_count_safe, gpu_select_devices,
// gpu_worker) existent dans les DEUX cas (CUDA ou non) avec les MÊMES signatures ne référençant
// que des types déjà connus de main(), afin que main() s'écrive de façon identique dans les deux
// compilations (g++ : toujours 0 GPU disponible, donc chemin CPU pur inchangé).
#if MEL_GPU
// U256 (hôte, 4 limbes 64 bits) <-> DU256 (device, 8 limbes 32 bits, voir commentaire détaillé au
// site de la nouvelle struct DU256) : chaque d[i] se scinde en v[2*i] (32 bits de poids faible) et
// v[2*i+1] (32 bits de poids fort), et réciproquement.
static inline DU256 toD(const U256 &x) {
    DU256 r;
    for (int i = 0; i < 4; i++) { r.v[2*i] = (u32)x.d[i]; r.v[2*i+1] = (u32)(x.d[i] >> 32); }
    return r;
}
static inline U256 fromD(const DU256 &x) {
    U256 r;
    for (int i = 0; i < 4; i++) r.d[i] = (u64)x.v[2*i] | ((u64)x.v[2*i+1] << 32);
    return r;
}
static inline DPoint toDP(const Point &p) { DPoint r; r.x=toD(p.x); r.y=toD(p.y); return r; }

static int gpu_device_count_safe() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess) ? n : 0;
}

// Table partagée des multiples 1..GPU_HALF d'un point de base arbitraire 'step' (affine),
// construite UNE FOIS sur CPU (aff_add, déjà testé) puis copiée sur chaque GPU utilisé.
// Indépendante de g_wt_main (table CPU, HALF=512). step = G pour le mode normal (clés
// consécutives) ; step = scalar_mul_fixed(2^start_bit) pour un segment du mode voisinage (les
// clés d'un segment ne diffèrent que par ses N bits, donc sautent de 2^start_bit en 2^start_bit).
static std::vector<DPoint> build_gpu_win_host(const Point &step) {
    std::vector<Point> pts(GPU_HALF);
    Point cur = step; pts[0] = cur;
    for (int i = 1; i < GPU_HALF; i++) { cur = aff_add(cur, step); pts[i] = cur; }
    std::vector<DPoint> out(GPU_HALF);
    for (int i = 0; i < GPU_HALF; i++) out[i] = toDP(pts[i]);
    return out;
}

// Tampon device RAII (libération automatique : l'auto-test a de nombreuses sorties anticipées).
template <class T> struct DevBuf {
    T *p = nullptr;
    DevBuf() {}
    DevBuf(const DevBuf &) = delete; DevBuf &operator=(const DevBuf &) = delete;
    ~DevBuf() { if (p) cudaFree(p); }
    bool alloc(size_t n) { return cudaMalloc(&p, (n ? n : 1) * sizeof(T)) == cudaSuccess; }
    bool up(const T *h, size_t n) { return cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess; }
    bool down(T *h, size_t n) const { return cudaMemcpy(h, p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess; }
};

// Auto-test GPU en DEUX étapes, sur GPU_SELFTEST_BLOCKS blocs de GPU_BT threads (tous coopérant par la
// mémoire partagée, comme en recherche réelle) :
//   1. gpu_selftest_kernel : chaque thread parcourt son job ; CHAQUE point et CHAQUE hash160 sont comparés à la
//      référence CPU (aff_add + hash160_bytes, déjà validés par self_test()) ;
//   2. gpu_scan_kernel (le vrai kernel de recherche) sur les MÊMES jobs, avec 3 cibles prises à des positions
//      délicates (début de 2e fenêtre, toute dernière clé d'un job, dernière clé d'un job d'une seule fenêtre) :
//      il doit remonter exactement ces 3 correspondances, avec le bon job, le bon décalage et la bonne cible.
// Un GPU qui échoue est exclu de la recherche (voir gpu_select_devices). 'step' est le même point de base que
// celui de 'win_host' : teste donc le mode normal (step=G) comme le mode voisinage (step=2^start_bit * G).
//
// Points de départ : surtout PAS 'step' lui-même (1*step) — il coïnciderait avec la première entrée de la table
// de fenêtre et déclencherait le cas dégénéré (delta nul) dès le premier pas : faux échec garanti. On part de points
// obtenus via scalar_mul_fixed sur des constantes arbitraires, exactement comme gpu_worker en conditions réelles.
static const int GPU_SELFTEST_BLOCKS = 2;
static const u128 GPU_SELFTEST_SCALAR = ((u128)0x9E3779B97F4A7C15ULL << 64) | 0xF39CC0605CEDC835ULL;
static inline u64 gpu_selftest_n(int i) {
    const u64 H = GPU_HALF;
    switch (i & 7) {
        case 0: return 0;
        case 1: return 1;
        case 2: return H - 1;
        case 3: return H;
        case 4: return H + 1;
        case 5: return 2 * H;
        case 6: return 3 * H + 7;
        default: return 2 * H + 3;
    }
}
static bool gpu_run_self_test(int device, const std::vector<DPoint> &win_host, const Point &step) {
    if (cudaSetDevice(device) != cudaSuccess) return false;
    const int NT = GPU_SELFTEST_BLOCKS * GPU_BT;
    const u64 H = GPU_HALF, STRIDE = 3 * H + 7; // = max de gpu_selftest_n
    std::vector<Point> st(NT);
    std::vector<GpuJob> jobs(NT);
    for (int i = 0; i < NT; i++) {
        u128 sc = GPU_SELFTEST_SCALAR + (u128)(i + 1) * (u128)0x1F2E3D4C5B6A7988ULL;
        st[i] = scalar_mul_fixed(u128_to_U256(sc));
        jobs[i].start = toDP(st[i]); jobs[i].n = gpu_selftest_n(i);
    }
    DevBuf<DPoint> d_win; DevBuf<GpuJob> d_jobs; DevBuf<DU256> d_x, d_hash; DevBuf<u64> d_par;
    if (!d_win.alloc(GPU_HALF) || !d_win.up(win_host.data(), GPU_HALF)) return false;
    if (!d_jobs.alloc(NT) || !d_jobs.up(jobs.data(), NT)) return false;
    if (!d_x.alloc((size_t)NT * STRIDE) || !d_par.alloc((size_t)NT * STRIDE) || !d_hash.alloc((size_t)NT * STRIDE)) return false;

    // ---- étape 1 : tous les points et hash160 ----
    GPU_LAUNCH(gpu_selftest_kernel, GPU_SELFTEST_BLOCKS, GPU_BT, nullptr, d_win.p, d_jobs.p, STRIDE, d_x.p, d_par.p, d_hash.p);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) return false;
    std::vector<DU256> hx((size_t)NT * STRIDE), hh((size_t)NT * STRIDE);
    std::vector<u64> hp((size_t)NT * STRIDE);
    if (!d_x.down(hx.data(), hx.size()) || !d_par.down(hp.data(), hp.size()) || !d_hash.down(hh.data(), hh.size())) return false;

    struct Pick { int job; u64 off; U256 h; };
    Pick picks[3] = { {6, H, ZERO}, {14, 3 * H + 6, ZERO}, {3, H - 1, ZERO} };
    for (int i = 0; i < NT; i++) {
        Point cur = st[i];
        for (u64 k = 0; k < jobs[i].n; k++) {
            if (k > 0) cur = aff_add(cur, step);
            const size_t ix = (size_t)i * STRIDE + k;
            if (!eq(fromD(hx[ix]), cur.x)) return false;
            if (hp[ix] != (cur.y.d[0] & 1ull)) return false;
            unsigned char comp[33], h20[20];
            point_to_compressed(cur, comp); hash160_bytes(comp, 33, h20);
            U256 ref = pack20(h20);
            if (!eq(fromD(hh[ix]), ref)) return false;
            for (auto &pk : picks) if (pk.job == i && pk.off == k) pk.h = ref;
        }
    }

    // ---- étape 2 : le vrai kernel de recherche doit retrouver exactement les 3 cibles ----
    std::vector<DU256> th(3); std::vector<u64> filt(1024, 0);
    for (int j = 0; j < 3; j++) {
        th[j] = toD(picks[j].h);
        u64 ix = picks[j].h.d[0] & 0xFFFF; filt[ix >> 6] |= 1ull << (ix & 63);
    }
    DevBuf<DU256> d_th; DevBuf<u64> d_filt; DevBuf<GpuHit> d_hits; DevBuf<int> d_hc;
    if (!d_th.alloc(3) || !d_th.up(th.data(), 3)) return false;
    if (!d_filt.alloc(1024) || !d_filt.up(filt.data(), 1024)) return false;
    if (!d_hits.alloc(64) || !d_hc.alloc(1)) return false;
    int zero = 0;
    if (!d_hc.up(&zero, 1)) return false;
    GPU_LAUNCH(gpu_scan_kernel, GPU_SELFTEST_BLOCKS, GPU_BT, nullptr, d_win.p, d_jobs.p, NT, d_th.p, 3, d_filt.p, d_hits.p, d_hc.p, 64);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) return false;
    int hc = 0; GpuHit gh[64];
    if (!d_hc.down(&hc, 1) || hc != 3 || !d_hits.down(gh, 3)) return false;
    for (int j = 0; j < 3; j++) {
        bool found = false;
        for (int q = 0; q < 3; q++)
            if (gh[q].job == picks[j].job && gh[q].offset == picks[j].off && gh[q].target == j) found = true;
        if (!found) return false;
    }
    return true;
}

// Liste des GPU détectés ET ayant réussi l'auto-test (voir gpu_run_self_test) ; un GPU qui échoue
// est simplement absent de la liste (le CPU, ou les autres GPU, prennent le relais). Teste à la
// fois un pas « mode normal » (step=G) et un pas « mode voisinage » (step=2^40 * G, représentatif
// d'un segment réel) : gpu_scan_device/gpu_selftest_kernel ne dépendent mathématiquement pas du
// point de base utilisé, mais ce double test couvre explicitement les deux usages.
static std::vector<int> gpu_select_devices() {
    std::vector<int> good;
    int n = gpu_device_count_safe();
    if (n <= 0) return good;
    std::vector<DPoint> win_g = build_gpu_win_host(G);
    Point step_nb = scalar_mul_fixed(u128_to_U256((u128)1 << 40));
    std::vector<DPoint> win_nb = build_gpu_win_host(step_nb);
    // Un thread par carte : les auto-tests (indépendants, tables en lecture seule, cudaSetDevice propre à chaque thread)
    // tournent en parallèle au lieu de s'enchaîner.
    std::vector<char> ok((size_t)n, 0);
    std::vector<std::thread> ths;
    for (int d = 0; d < n; d++)
        ths.emplace_back([&, d] {
            bool ok1 = gpu_run_self_test(d, win_g, G);
            ok[(size_t)d] = (ok1 && gpu_run_self_test(d, win_nb, step_nb)) ? 1 : 0;
        });
    for (auto &t : ths) t.join();
    for (int d = 0; d < n; d++) {
        if (ok[(size_t)d]) good.push_back(d);
        else std::cerr << "Note : auto-test GPU echoue pour le GPU " << d << " ; il est ignore.\n";
    }
    return good;
}

// Construit les tableaux "plats" de la TargetSet pour upload GPU (hash160 empaquetés + filtre).
static void build_target_arrays_host(std::vector<DU256> &th, std::vector<u64> &filt) {
    th.resize(g_ts.size());
    for (size_t i = 0; i < g_ts.size(); i++) th[i] = toD(g_ts.h[i]);
    filt.assign(1024, 0);
    for (int i = 0; i < 1024; i++) filt[i] = g_ts.filter[i];
}

// Prépare un lot de 'count' jobs en PARALLÈLE sur plusieurs threads CPU, par SOUS-PLAGES disjointes [a,b) :
// fn(a, b) remplit les jobs a..b-1 (écritures uniquement dans les éléments d'indice k de ce sous-intervalle,
// aucune synchronisation nécessaire). Une sous-plage permet à fn d'amortir un travail commun (une inversion
// modulaire pour toute la sous-plage, une progression arithmétique de points...) au lieu d'un calcul isolé par
// job. scalar_mul_fixed/jac_madd sont des fonctions pures (g_gtable figée après le démarrage).
// 'nthreads' vient du paramètre -t déjà existant.
template <class Fn>
static void prepare_ranges_parallel(uint64_t count, unsigned nthreads, Fn &&fn) {
    if (count == 0) return;
    uint64_t nth = std::max<uint64_t>(1, std::min<uint64_t>(nthreads, count));
    if (nth == 1) { fn((uint64_t)0, count); return; }
    std::vector<std::thread> ths;
    ths.reserve((size_t)nth);
    uint64_t chunk = (count + nth - 1) / nth;
    for (uint64_t ti = 0; ti < nth; ti++) {
        uint64_t a = ti * chunk, b = std::min(count, a + chunk);
        if (a >= b) break;
        ths.emplace_back([a, b, &fn] { fn(a, b); });
    }
    for (auto &th : ths) th.join();
}

// Taille de grille (nombre de blocs) choisie AUTOMATIQUEMENT pour une carte : nombre de SM x blocs résidents par SM
// (occupation réelle du kernel de recherche, avec GPU_BT threads par bloc) x 8 vagues, borné à [SM, 4096].
// Utilisée quand --gpu-blocks n'est pas donné : une grande carte reçoit une grille plus large qu'une petite, donc
// chaque lancement dure à peu près le même temps quel que soit le GPU (arrêt Ctrl-C réactif, CPU/GPU bien recouverts).
static int gpu_auto_blocks(int device) {
#if defined(__CUDACC__)
    int sms = 0, occ = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms <= 0) return 1024;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, gpu_scan_kernel, GPU_BT, 0) != cudaSuccess || occ <= 0) occ = GPU_MINBLK;
    int b = sms * occ * 8;
    return std::max(sms, std::min(b, 4096));
#else
    (void)device; return 1024;
#endif
}

// Attend la fin du travail d'un stream SANS tourner en boucle active : un événement CUDA « blocking sync » met le thread
// hôte en sommeil jusqu'à la fin du kernel (cudaStreamSynchronize, lui, fait du spin et occupe 100 % d'un coeur CPU,
// ce qui volerait du temps aux workers CPU du mode hybride). Équivalent à cudaStreamSynchronize pour la gestion d'erreur.
static cudaError_t gpu_wait_stream(cudaStream_t st) {
#if defined(__CUDACC__)
    cudaEvent_t ev;
    if (cudaEventCreateWithFlags(&ev, cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess)
        return cudaStreamSynchronize(st);
    cudaError_t e = cudaEventRecord(ev, st);
    if (e == cudaSuccess) e = cudaEventSynchronize(ev);
    cudaEventDestroy(ev);
    return e;
#else
    return cudaStreamSynchronize(st);
#endif
}

// Largement supérieur au nombre de correspondances réelles possibles en un seul lancement de
// kernel (au plus g_ts.size(), <= 80) : marge très confortable, jamais atteinte en pratique.
static const int GPU_MAX_HITS = 4096;

// Thread hôte piloté par un GPU : revendique des PLAGES de chunks (claim_range), construit les
// jobs correspondants EXACTEMENT comme worker() le fait chunk par chunk (même calcul d'offset,
// même scalar_mul_fixed déjà testé), les envoie au GPU, puis revérifie et enregistre chaque
// correspondance potentielle via handle_hit() (identique au chemin CPU : verify_key() recalcule
// tout depuis zéro avant tout enregistrement).
//
// DOUBLE TAMPON (2 jeux de buffers device + 2 streams CUDA) : pendant que le kernel du lancement
// COURANT tourne sur le GPU (lancé de façon asynchrone, sans attendre sa fin), le CPU prépare déjà
// en PARALLÈLE les jobs du lancement SUIVANT (prepare_jobs_parallel, sur plusieurs threads). Sans
// ce recouvrement, le GPU restait totalement inactif pendant toute la préparation CPU de CHAQUE
// lancement ; ce recouvrement masque ce temps CPU derrière l'exécution GPU du lancement précédent
// (tant que celle-ci dure au moins aussi longtemps, ce qui est quasiment toujours le cas : un
// kernel traite des centaines de milliers de clés par thread pendant que la préparation ne fait
// qu'UN scalar_mul_fixed par job).
//
// Contrairement à claim()/claim_range() (pensés pour UNE SEULE plage revendiquée et non terminée à
// la fois par thread), on a ici jusqu'à DEUX plages simultanément "en vol" : celle qui tourne sur
// le GPU et celle que le CPU est en train de préparer/vient de lancer. g_cur[t] est donc remis à
// jour manuellement (update_progress_marker) pour pointer TOUJOURS sur le début de la plus
// ANCIENNE des deux tant qu'elle n'est pas confirmée terminée (son stream synchronisé) : ainsi
// safe_index()/all_idle() (utilisés par main() pour l'affichage et pour savoir quand joindre les
// threads) ne voient jamais ce thread "plus avancé" qu'il ne l'est réellement.
// Conséquence sur Ctrl-C : dans le pire cas (arrêt demandé juste après le lancement du tampon
// suivant), il faut attendre la fin de CE lancement-là en plus de celui déjà en cours — au plus un
// lancement GPU de plus qu'avant, chacun toujours borné par la taille de grille (blocks x threads).
static void gpu_worker(unsigned t, int device, u128 lo, u128 span, int cb, uint64_t stride, int blocks, int threads, unsigned nt) {
    if (cudaSetDevice(device) != cudaSuccess) { release(t); return; }
    if (blocks <= 0) blocks = gpu_auto_blocks(device); // grille automatique propre à cette carte
    std::vector<DPoint> win_host = build_gpu_win_host(G);
    std::vector<DU256> th; std::vector<u64> filt;
    build_target_arrays_host(th, filt);

    DPoint *d_win=nullptr; DU256 *d_th=nullptr; u64 *d_filt=nullptr;
    GpuJob *d_jobs[2] = {nullptr, nullptr};
    GpuHit *d_hits[2] = {nullptr, nullptr};
    int *d_hitcount[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {nullptr, nullptr};
    size_t want = (size_t)blocks * (size_t)threads;
    bool ok = true;
    ok = ok && cudaMalloc(&d_win, GPU_HALF*sizeof(DPoint)) == cudaSuccess;
    ok = ok && cudaMemcpy(d_win, win_host.data(), GPU_HALF*sizeof(DPoint), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMalloc(&d_th, std::max<size_t>(1,th.size())*sizeof(DU256)) == cudaSuccess;
    if (ok && !th.empty()) ok = cudaMemcpy(d_th, th.data(), th.size()*sizeof(DU256), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMalloc(&d_filt, 1024*sizeof(u64)) == cudaSuccess;
    ok = ok && cudaMemcpy(d_filt, filt.data(), 1024*sizeof(u64), cudaMemcpyHostToDevice) == cudaSuccess;
    for (int s = 0; s < 2 && ok; s++) {
        ok = ok && cudaMalloc(&d_jobs[s], want*sizeof(GpuJob)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hits[s], GPU_MAX_HITS*sizeof(GpuHit)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hitcount[s], sizeof(int)) == cudaSuccess;
        ok = ok && cudaStreamCreate(&stream[s]) == cudaSuccess;
    }
    if (!ok) {
        std::cerr << "Erreur : allocation memoire GPU " << device << " echouee ; ce GPU est ignore.\n";
        if (d_win) cudaFree(d_win); if (d_th) cudaFree(d_th); if (d_filt) cudaFree(d_filt);
        for (int s = 0; s < 2; s++) {
            if (d_jobs[s]) cudaFree(d_jobs[s]); if (d_hits[s]) cudaFree(d_hits[s]); if (d_hitcount[s]) cudaFree(d_hitcount[s]);
            if (stream[s]) cudaStreamDestroy(stream[s]);
        }
        release(t);
        return;
    }

    const uint64_t n_full = 1ULL << cb, mask = g_nchunks - 1;

    // État de chaque tampon (0 et 1, utilisés en alternance). Les vecteurs sont alloués une seule
    // fois ici et réutilisés (écrasés) à chaque lancement de ce tampon, comme dans la version
    // précédente (un seul tampon).
    struct Slot {
        std::vector<GpuJob> jobs;
        std::vector<u128> job_start_key;
        std::vector<uint64_t> job_n;
        std::vector<GpuHit> hits;
        uint64_t start_idx = 0, count = 0;
        int hc = 0;          // nombre de correspondances renvoyées par le GPU pour ce lancement
        bool active = false; // lancement soumis et pas encore clos (finish() pas encore appelé)
    };
    Slot slot[2];
    for (int s = 0; s < 2; s++) {
        slot[s].jobs.resize(want); slot[s].job_start_key.resize(want); slot[s].job_n.resize(want);
        slot[s].hits.resize(GPU_MAX_HITS);
    }

    // Remplit (CPU, en parallèle) puis lance (GPU, asynchrone) le tampon 's' pour la plage
    // [start_idx, start_idx+count). NE SYNCHRONISE JAMAIS : revient dès que le kernel est soumis
    // (hors durée des petites copies H2D elles-mêmes), sans attendre son exécution.
    auto fill_and_launch = [&](int s, uint64_t start_idx, uint64_t count) {
        Slot &sl = slot[s];
        sl.start_idx = start_idx; sl.count = count; sl.active = true;
        prepare_ranges_parallel(count, nt, [&](uint64_t a, uint64_t b) {
            // Points de départ des chunks start_idx+a .. start_idx+b-1 : progression arithmétique (voir
            // build_progression_starts), une addition par job + une inversion pour toute la sous-plage.
            std::vector<Point> pts((size_t)(b - a));
            build_progression_starts(lo, cb, stride, g_nchunks, start_idx + a, b - a, pts.data());
            for (uint64_t k = a; k < b; k++) {
                uint64_t idx = start_idx + k;
                u128 offset = (u128)((idx * stride) & mask) << cb;
                if (offset >= span) { sl.job_n[k] = 0; sl.jobs[k].n = 0; continue; } // chunk hors plage réelle
                uint64_t n = (uint64_t)std::min<u128>((u128)n_full, span - offset);
                u128 start = lo + offset;
                sl.job_start_key[k] = start; sl.job_n[k] = n;
                sl.jobs[k].start = toDP(pts[(size_t)(k - a)]); sl.jobs[k].n = n;
            }
        });
        for (uint64_t k = count; k < want; k++) sl.jobs[k].n = 0; // threads CUDA surnuméraires : rien à faire

        cudaMemcpyAsync(d_jobs[s], sl.jobs.data(), want*sizeof(GpuJob), cudaMemcpyHostToDevice, stream[s]);
        int zero = 0;
        cudaMemcpyAsync(d_hitcount[s], &zero, sizeof(int), cudaMemcpyHostToDevice, stream[s]);
        GPU_LAUNCH(gpu_scan_kernel, blocks, threads, stream[s], d_win, d_jobs[s], (int)want, d_th, (int)th.size(),
                   d_filt, d_hits[s], d_hitcount[s], GPU_MAX_HITS);
        // Le compteur de correspondances est relu plus tard dans finish(), APRÈS synchronisation du
        // stream : le lire ici en asynchrone forcerait de toute façon le CPU à attendre la fin du
        // kernel (copie D2H ordonnée après lui dans le même stream, vers de la mémoire hôte non
        // verrouillée), ce qui annulerait exactement le recouvrement recherché.
    };

    // Attend la fin du lancement du tampon 's' (synchronise son stream), traite ses résultats
    // (compteur, correspondances via handle_hit, comptage de clés) puis le marque libre.
    // Retourne false en cas d'erreur CUDA fatale (ce GPU doit alors être arrêté).
    auto finish = [&](int s) -> bool {
        Slot &sl = slot[s];
        cudaError_t kerr = gpu_wait_stream(stream[s]);
        if (kerr == cudaSuccess) kerr = cudaGetLastError(); // erreur de LANCEMENT (config. invalide, trop de ressources...)
        if (kerr != cudaSuccess) {
            std::cerr << "Erreur CUDA (GPU " << device << ") : " << cudaGetErrorString(kerr) << " ; ce GPU est arrete.\n";
            g_gpu_lost = true;
            sl.active = false;
            return false;
        }
        cudaMemcpy(&sl.hc, d_hitcount[s], sizeof(int), cudaMemcpyDeviceToHost);
        if (sl.hc > 0) {
            int ncopy = std::min(sl.hc, GPU_MAX_HITS);
            cudaMemcpy(sl.hits.data(), d_hits[s], ncopy*sizeof(GpuHit), cudaMemcpyDeviceToHost);
            for (int i = 0; i < ncopy; i++) {
                uint64_t jk = (uint64_t)sl.hits[i].job;
                if (jk < sl.count) handle_hit(sl.job_start_key[jk] + (u128)sl.hits[i].offset, sl.hits[i].target);
            }
        }
        uint64_t total_n = 0; for (uint64_t k = 0; k < sl.count; k++) total_n += sl.job_n[k];
        g_count.fetch_add(total_n, std::memory_order_relaxed);
        g_count_gpu.fetch_add(total_n, std::memory_order_relaxed);
        sl.active = false;
        return true;
    };

    // g_cur[t] doit toujours pointer sur le début de la plage la plus ANCIENNE encore en vol (voir
    // commentaire au-dessus de la fonction). 'oldest' = indice de tampon encore actif le plus
    // ancien, ou -1 si plus aucun tampon n'est en vol.
    auto update_progress_marker = [&](int oldest) {
        std::lock_guard<std::mutex> l(g_mtx);
        g_cur[t] = (oldest >= 0) ? slot[oldest].start_idx : IDLE;
    };

    int cur = 0;
    uint64_t s_idx = 0, cnt = 0;
    bool have_cur = !g_stop.load(std::memory_order_relaxed) && claim_range(t, (uint64_t)want, s_idx, cnt);
    if (have_cur) fill_and_launch(cur, s_idx, cnt); // claim_range a déjà positionné g_cur[t]=s_idx, correct ici

    bool fatal = false;
    while (have_cur && !fatal) {
        int nxt = 1 - cur;
        bool have_nxt = false;
        if (!g_stop.load(std::memory_order_relaxed)) {
            uint64_t ns_idx, ncnt;
            if (claim_range(t, (uint64_t)want, ns_idx, ncnt)) {
                // claim_range vient d'écraser g_cur[t] avec ns_idx (plus récent) ; on le ramène au
                // début de 'cur' (plus ancien, encore en vol) avant de préparer/lancer 'nxt'.
                update_progress_marker(cur);
                fill_and_launch(nxt, ns_idx, ncnt); // CPU+GPU recouvrent l'exécution encore en cours de 'cur'
                have_nxt = true;
            }
        }
        fatal = !finish(cur);
        if (fatal && have_nxt) {
            // 'cur' vient d'échouer alors que 'nxt' était déjà soumis au GPU DANS CETTE MÊME
            // itération (avant qu'on s'aperçoive de l'échec de 'cur') : on le draine au mieux —
            // sans chercher à exploiter ses résultats, le GPU étant déjà jugé défaillant — avant
            // de libérer quoi que ce soit, pour ne jamais faire de cudaFree sous un kernel encore
            // actif. On ignore volontairement une éventuelle seconde erreur ici : on est déjà en
            // train d'arrêter ce GPU.
            cudaStreamSynchronize(stream[nxt]);
            slot[nxt].active = false;
        }
        update_progress_marker((!fatal && have_nxt) ? nxt : -1);
        cur = nxt;
        have_cur = have_nxt;
    }
    // Garde-fou : à ce stade, le tampon 'cur' courant a toujours déjà été clos (par le finish(cur)
    // ci-dessus, ou par le drainage de secours juste au-dessus si 'cur' est fatal) — la boucle ne
    // réaffecte 'cur' qu'après coup. Ne devrait donc jamais s'exécuter ; conservé par prudence pour
    // ne jamais laisser un stream non synchronisé derrière avant les cudaFree/cudaStreamDestroy.
    if (slot[cur].active) { cudaStreamSynchronize(stream[cur]); slot[cur].active = false; update_progress_marker(-1); }

    cudaFree(d_win); cudaFree(d_th); cudaFree(d_filt);
    for (int s = 0; s < 2; s++) {
        cudaFree(d_jobs[s]); cudaFree(d_hits[s]); cudaFree(d_hitcount[s]); cudaStreamDestroy(stream[s]);
    }
}
#else
static int gpu_device_count_safe() { return 0; }
static std::vector<int> gpu_select_devices() { return {}; }
static void gpu_worker(unsigned t, int, u128, u128, int, uint64_t, int, int, unsigned) { release(t); }
#endif // MEL_GPU

// =================== Mode voisinage (-v / -vv / -vvv / -vvvv) ===================
struct Segment { int start_bit; int len; const char *nom; };
// free_bits = nombre de bits qui varient réellement dans la plage (voir run_neighbor_mode) : pour
// une plage classique "bits" ([2^(bits-1), 2^bits-1]), free_bits == bits - 1 comme avant ; pour une
// plage personnalisée "début:fin", free_bits dépend de la largeur réelle de la plage.
static bool segment_fits(int free_bits, int N) { return N >= 1 && N <= free_bits; }
static std::vector<Segment> build_segments(int level, int free_bits, int N) {
    Segment prefix{ free_bits - N, N, "prefixe" };
    Segment suffix{ 0, N, "suffixe" };
    Segment middle{ (free_bits - N) / 2, N, "milieu" };
    switch (level) {
        case 1: return {prefix};
        case 2: return {middle};
        case 3: return {suffix};
        default: return {prefix, middle, suffix};
    }
}

// ---- Ordre de visite des cellules : permutation pseudo-aléatoire SANS DOUBLON ----
// Le mode voisinage découpe l'intervalle en « cellules » : pour un segment de N bits, une cellule est
// un motif de (free_bits - N) bits (les bits hors segment) ; l'essai de cette cellule balaie les 2^N
// clés qui ne diffèrent que par les bits du segment. Les 2^(free_bits-N) cellules d'un segment
// partitionnent donc l'intervalle ENTIER, sans recouvrement.
// Pour visiter ces cellules dans un ordre aléatoire mais en SACHANT exactement lesquelles ont déjà
// été faites, on parcourt les indices 0,1,2,... d'un compteur atomique et on les envoie par une
// permutation pseudo-aléatoire de [0, 2^k) (réseau de Feistel à clés tirées au hasard à chaque
// lancement, avec « cycle-walking » pour tomber exactement dans [0, 2^k)). Une permutation est une
// bijection : aucune cellule n'est jamais refaite, aucune n'est oubliée, et il suffit de retenir le
// compteur (zéro mémoire, zéro verrou) pour connaître l'état d'avancement et détecter la fin.
struct CellPerm {
    static const int ROUNDS = 6;
    int k = 0, h = 0; u128 total = 1; u64 mask = 0; u64 rk[ROUNDS] = {};
    static inline u64 mix(u64 z) { // splitmix64
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    void init(int kk, std::mt19937_64 &g) {
        k = kk; total = (u128)1 << k; h = (k + 1) / 2;
        mask = (h >= 64) ? ~0ULL : ((1ULL << h) - 1);
        for (int r = 0; r < ROUNDS; r++) rk[r] = g();
    }
    // x dans [0, 2^k)  ->  image dans [0, 2^k) ; bijective.
    u128 apply(u128 x) const {
        if (k == 0) return 0;
        do { // Feistel sur 2h >= k bits ; on recommence tant que le résultat sort de [0, 2^k) (< 2 tours en moyenne)
            u64 L = (u64)(x >> h) & mask, R = (u64)x & mask;
            for (int r = 0; r < ROUNDS; r++) { u64 f = L ^ (mix(R ^ rk[r]) & mask); L = R; R = f; }
            x = ((u128)L << h) | R;
        } while (x >= total);
        return x;
    }
};
// Décalage (par rapport à 'lo') du début de la cellule c pour ce segment : on insère N bits à zéro
// à la position start_bit dans le motif c de (free_bits - N) bits.
static inline u128 cell_to_off(u128 c, const Segment &seg) {
    u128 low = c & (((u128)1 << seg.start_bit) - 1);
    return ((c >> seg.start_bit) << (seg.start_bit + seg.len)) | low;
}

static std::atomic<uint64_t> g_nb_next{0};      // prochain indice de cellule à réserver (= essais lancés)
static std::atomic<bool> g_nb_abandoned{false}; // un thread a quitté un essai en cours (arrêt/Ctrl-C/clé trouvée)

// Corps d'un thread du mode voisinage. Chaque thread réserve l'indice suivant (fetch_add, sans
// verrou), en déduit sa cellule par la permutation, puis balaie TOUT SEUL les segments de cet essai
// avant de passer au suivant : les nt threads travaillent sur des essais indépendants en parallèle,
// donc tous les cœurs restent occupés quelle que soit la taille des segments.
// Aux niveaux -vvvv (3 segments) le même motif c sert aux trois segments (N bits insérés en préfixe,
// au milieu puis en suffixe) : chaque type de segment parcourt ainsi lui aussi l'intervalle sans doublon.
static void neighbor_worker(const CellPerm &perm, u128 total_cells, u128 lo, u128 span,
                             const std::vector<Segment> &segments, const std::vector<WinTable> &seg_wt) {
    Scratch sc;
    auto must_stop = []() { return g_stop.load(std::memory_order_relaxed) || g_found.load(std::memory_order_relaxed); };
    while (!must_stop()) {
        uint64_t idx = g_nb_next.fetch_add(1, std::memory_order_relaxed);
        if ((u128)idx >= total_cells) break; // toutes les cellules ont été réservées : plage épuisée
        u128 c = perm.apply((u128)idx);
        bool abandoned = false;
        for (size_t i = 0; i < segments.size(); ++i) {
            if (must_stop()) { abandoned = true; break; }
            const Segment &seg = segments[i];
            u128 off = cell_to_off(c, seg);
            if (off >= span) continue; // cellule entièrement hors plage (plage non puissance de 2) : rien à scanner
            u128 base = lo + off;
            Point start_pt = scalar_mul_fixed(u128_to_U256(base));
            uint64_t n = (uint64_t)1 << seg.len;
            int64_t r = scan_batch(seg_wt[i], start_pt, n, sc, g_count, g_ts, must_stop,
                                   [&](int64_t hit, int ti) { handle_hit(base + ((u128)hit << seg.start_bit), ti); });
            if (r == -2) { abandoned = true; break; }
        }
        if (abandoned) { g_nb_abandoned = true; break; }
    }
}

// ---- Équivalent GPU de neighbor_worker ----
// Réutilise TEL QUEL le moteur de scan GPU déjà écrit et testé pour le mode normal (GpuJob,
// GpuHit, gpu_scan_kernel, gpu_scan_device) : un « job » GPU est générique (point de départ +
// nombre de clés à scanner avec une table de fenêtre donnée), il ne sait pas s'il représente un
// morceau de plage contiguë (mode normal) ou un segment de voisinage (mode voisinage). Seule la
// table de fenêtre change : au lieu de multiples de G, on utilise les multiples de
// 2^start_bit * G propres à chaque segment (seg_gpu_win, une par segment, calculées une seule
// fois dans run_neighbor_mode). Avec le niveau -vvvv (3 segments), les 3 segments d'un même lot de
// cellules sont traités par 3 lancements de kernel successifs (un par segment), chacun avec sa
// propre table de fenêtre.
// DOUBLE TAMPON (mêmes principes que gpu_worker, voir son commentaire détaillé) : la séquence des
// lancements est ici (lot de cellules x segment) — avec -vvvv (3 segments), un même lot de cellules
// donne 3 lancements successifs (un par segment), avant de passer au lot suivant. On la traite
// comme une file linéaire de "rounds" (struct Round ci-dessous) : le round suivant est soit "même
// lot, segment suivant" (aucune réservation supplémentaire), soit "nouveau lot, segment 0" (réserve
// via g_nb_next.fetch_add). Contrairement à gpu_worker, aucune comptabilité façon
// claim_range/g_cur/safe_index n'existe ici : la progression affichée (run_neighbor_mode) lit
// directement g_nb_next (cellules RÉSERVÉES, pas forcément confirmées scannées, déjà le cas avant
// ce changement), donc rien à adapter de ce côté. En revanche g_nb_abandoned (qui conditionne le
// message final "plage entièrement scannée") doit rester fidèle à l'original : il doit passer à
// true dès qu'un lot de cellules réservé n'a pas vu TOUS ses segments traités (arrêt demandé ou
// erreur CUDA en cours de route), exactement comme le faisait la version à un seul tampon.
#if MEL_GPU
static void gpu_neighbor_worker(int device, u128 lo, u128 span,
                                 const std::vector<Segment> &segments,
                                 const std::vector<std::vector<DPoint>> &seg_win_host,
                                 const CellPerm &perm, u128 total_cells,
                                 int blocks, int threads, unsigned nt) {
    if (cudaSetDevice(device) != cudaSuccess) { g_nb_abandoned = true; return; }
    if (blocks <= 0) blocks = gpu_auto_blocks(device); // grille automatique propre à cette carte
    size_t nseg = segments.size();
    std::vector<DPoint*> d_win(nseg, nullptr);
    std::vector<DU256> th; std::vector<u64> filt;
    build_target_arrays_host(th, filt);
    DU256 *d_th = nullptr; u64 *d_filt = nullptr;
    GpuJob *d_jobs[2] = {nullptr, nullptr};
    GpuHit *d_hits[2] = {nullptr, nullptr};
    int *d_hitcount[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {nullptr, nullptr};
    size_t want = (size_t)blocks * (size_t)threads;

    bool ok = true;
    for (size_t s = 0; s < nseg; s++) {
        ok = ok && cudaMalloc(&d_win[s], GPU_HALF*sizeof(DPoint)) == cudaSuccess;
        ok = ok && cudaMemcpy(d_win[s], seg_win_host[s].data(), GPU_HALF*sizeof(DPoint), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    ok = ok && cudaMalloc(&d_th, std::max<size_t>(1,th.size())*sizeof(DU256)) == cudaSuccess;
    if (ok && !th.empty()) ok = cudaMemcpy(d_th, th.data(), th.size()*sizeof(DU256), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMalloc(&d_filt, 1024*sizeof(u64)) == cudaSuccess;
    ok = ok && cudaMemcpy(d_filt, filt.data(), 1024*sizeof(u64), cudaMemcpyHostToDevice) == cudaSuccess;
    for (int s = 0; s < 2 && ok; s++) {
        ok = ok && cudaMalloc(&d_jobs[s], want*sizeof(GpuJob)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hits[s], GPU_MAX_HITS*sizeof(GpuHit)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_hitcount[s], sizeof(int)) == cudaSuccess;
        ok = ok && cudaStreamCreate(&stream[s]) == cudaSuccess;
    }

    if (!ok) {
        std::cerr << "Erreur : allocation memoire GPU " << device << " echouee (voisinage) ; ce GPU est ignore.\n";
    } else {
        auto must_stop = []() { return g_stop.load(std::memory_order_relaxed) || g_found.load(std::memory_order_relaxed); };

        // Un "round" = un lancement = un lot de cellules [start_idx, start_idx+count) scanné avec
        // la table de fenêtre du segment 'seg'.
        struct Round { uint64_t start_idx = 0, count = 0; size_t seg = 0; bool valid = false; };
        struct Slot {
            std::vector<GpuJob> jobs;
            std::vector<u128> job_base;
            std::vector<GpuHit> hits;
            Round r;
            int hc = 0;
            bool active = false;
        };
        Slot slot[2];
        for (int s = 0; s < 2; s++) {
            slot[s].jobs.resize(want); slot[s].job_base.resize(want); slot[s].hits.resize(GPU_MAX_HITS);
        }

        // Round suivant dans la file linéaire : même lot (segment suivant) si 'prev' n'a pas
        // encore traité son dernier segment, sinon nouveau lot (réservation via g_nb_next).
        // N'effectue AUCUNE vérification de must_stop() (à la charge de l'appelant).
        auto advance_or_claim = [&](const Round &prev, bool have_prev) -> Round {
            if (have_prev && prev.seg + 1 < nseg) return Round{ prev.start_idx, prev.count, prev.seg + 1, true };
            uint64_t start_idx = g_nb_next.fetch_add((uint64_t)want, std::memory_order_relaxed);
            if ((u128)start_idx >= total_cells) return Round{};
            uint64_t count = (uint64_t)std::min<u128>((u128)want, total_cells - (u128)start_idx);
            return Round{ start_idx, count, 0, true };
        };

        auto fill_and_launch = [&](int s, const Round &r) {
            Slot &sl = slot[s];
            sl.r = r; sl.active = true;
            const Segment &seg = segments[r.seg];
            uint64_t n_seg = (uint64_t)1 << seg.len;
            prepare_ranges_parallel(r.count, nt, [&](uint64_t a, uint64_t b) {
                // Cellules arbitraires (permutation de Feistel) : pas de progression arithmétique possible,
                // mais k*G en Jacobien puis UNE SEULE inversion pour toute la sous-plage (jac_batch_to_aff).
                std::vector<Jac> J((size_t)(b - a));
                std::vector<Point> pts((size_t)(b - a));
                for (uint64_t k = a; k < b; k++) {
                    u128 c = perm.apply((u128)(r.start_idx + k));
                    u128 off = cell_to_off(c, seg);
                    if (off >= span) { sl.jobs[k].n = 0; J[(size_t)(k - a)] = Jac{ZERO, ZERO, ZERO, true}; continue; } // hors plage
                    u128 base = lo + off;
                    sl.job_base[k] = base;
                    J[(size_t)(k - a)] = scalar_mul_fixed_jac(u128_to_U256(base));
                    sl.jobs[k].n = n_seg;
                }
                std::vector<U256> scratch;
                jac_batch_to_aff(J.data(), pts.data(), (size_t)(b - a), scratch);
                for (uint64_t k = a; k < b; k++) if (sl.jobs[k].n > 0) sl.jobs[k].start = toDP(pts[(size_t)(k - a)]);
            });
            for (uint64_t k = r.count; k < want; k++) sl.jobs[k].n = 0; // threads CUDA surnuméraires

            cudaMemcpyAsync(d_jobs[s], sl.jobs.data(), want*sizeof(GpuJob), cudaMemcpyHostToDevice, stream[s]);
            int zero = 0;
            cudaMemcpyAsync(d_hitcount[s], &zero, sizeof(int), cudaMemcpyHostToDevice, stream[s]);
            GPU_LAUNCH(gpu_scan_kernel, blocks, threads, stream[s], d_win[r.seg], d_jobs[s], (int)want, d_th,
                       (int)th.size(), d_filt, d_hits[s], d_hitcount[s], GPU_MAX_HITS);
        };

        auto finish = [&](int s) -> bool {
            Slot &sl = slot[s];
            cudaError_t kerr = gpu_wait_stream(stream[s]);
            if (kerr == cudaSuccess) kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                std::cerr << "Erreur CUDA (GPU " << device << ", voisinage) : " << cudaGetErrorString(kerr) << " ; ce GPU est arrete.\n";
                g_gpu_lost = true;
                sl.active = false;
                return false;
            }
            cudaMemcpy(&sl.hc, d_hitcount[s], sizeof(int), cudaMemcpyDeviceToHost);
            const Segment &seg = segments[sl.r.seg];
            if (sl.hc > 0) {
                int ncopy = std::min(sl.hc, GPU_MAX_HITS);
                cudaMemcpy(sl.hits.data(), d_hits[s], ncopy*sizeof(GpuHit), cudaMemcpyDeviceToHost);
                for (int i = 0; i < ncopy; i++) {
                    uint64_t jk = (uint64_t)sl.hits[i].job;
                    if (jk < sl.r.count && sl.jobs[jk].n > 0)
                        handle_hit(sl.job_base[jk] + ((u128)sl.hits[i].offset << seg.start_bit), sl.hits[i].target);
                }
            }
            uint64_t total_n = 0; for (uint64_t k = 0; k < sl.r.count; k++) if (sl.jobs[k].n > 0) total_n += sl.jobs[k].n;
            g_count.fetch_add(total_n, std::memory_order_relaxed);
            g_count_gpu.fetch_add(total_n, std::memory_order_relaxed);
            sl.active = false;
            return true;
        };

        int cur = 0;
        Round cur_r;
        bool have_cur = false;
        if (!must_stop()) {
            cur_r = advance_or_claim(Round{}, false);
            if (cur_r.valid) { fill_and_launch(cur, cur_r); have_cur = true; }
        }

        bool fatal = false;
        while (have_cur && !fatal) {
            int nxt = 1 - cur;
            bool have_nxt = false;
            Round nxt_r;
            bool batch_had_more_segments = (cur_r.seg + 1 < nseg);
            if (!must_stop()) {
                nxt_r = advance_or_claim(cur_r, true);
                if (nxt_r.valid) { fill_and_launch(nxt, nxt_r); have_nxt = true; }
            } else if (batch_had_more_segments) {
                // Le lot courant avait encore des segments à traiter : arrêt demandé avant qu'ils
                // le soient, ce lot n'est donc pas entièrement scanné.
                g_nb_abandoned = true;
            }
            fatal = !finish(cur);
            if (fatal) {
                g_nb_abandoned = true; // comme dans la version à un seul tampon : toute erreur CUDA fatale -> abandon
                if (have_nxt) { cudaStreamSynchronize(stream[nxt]); slot[nxt].active = false; } // drainage avant cudaFree
            }
            cur = nxt; cur_r = nxt_r; have_cur = have_nxt;
        }
        // Garde-fou (ne devrait jamais s'exécuter, voir gpu_worker) : ne jamais libérer sous un
        // kernel encore actif.
        if (slot[cur].active) { cudaStreamSynchronize(stream[cur]); slot[cur].active = false; }
    }
    for (size_t s = 0; s < nseg; s++) if (d_win[s]) cudaFree(d_win[s]);
    if (d_th) cudaFree(d_th); if (d_filt) cudaFree(d_filt);
    for (int s = 0; s < 2; s++) {
        if (d_jobs[s]) cudaFree(d_jobs[s]); if (d_hits[s]) cudaFree(d_hits[s]); if (d_hitcount[s]) cudaFree(d_hitcount[s]);
        if (stream[s]) cudaStreamDestroy(stream[s]);
    }
}
#else
static void gpu_neighbor_worker(int, u128, u128, const std::vector<Segment>&,
                                 const std::vector<std::vector<Point>>&, const CellPerm&, u128, int, int, unsigned) {}
#endif // MEL_GPU

// Enveloppe bloquante : construit la table de fenêtre GPU de chaque segment (2^start_bit * G,
// via scalar_mul_fixed déjà testé) puis lance et attend (join) tous les threads GPU. Cette
// fonction existe (avec la MÊME signature) dans les deux compilations afin que run_neighbor_mode
// n'ait jamais à nommer le type DPoint (qui n'existe que sous __CUDACC__) ; la variante non-CUDA
// ne fait rien (jamais appelée, puisque good_gpus est alors toujours vide).
#if MEL_GPU
static void run_neighbor_gpu_blocking(const std::vector<int> &good_gpus, u128 lo, u128 span,
                                       const std::vector<Segment> &segments, const CellPerm &perm,
                                       u128 total_cells, int gpu_blocks, int gpu_threads, unsigned nt) {
    std::vector<std::vector<DPoint>> seg_win_host(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        Point incPt = scalar_mul_fixed(u128_to_U256((u128)1 << segments[i].start_bit));
        seg_win_host[i] = build_gpu_win_host(incPt);
    }
    std::vector<std::thread> th;
    for (int d : good_gpus)
        th.emplace_back(gpu_neighbor_worker, d, lo, span, std::cref(segments), std::cref(seg_win_host),
                         std::cref(perm), total_cells, gpu_blocks, gpu_threads, nt);
    for (auto &t : th) t.join();
}
#else
static void run_neighbor_gpu_blocking(const std::vector<int>&, u128, u128, const std::vector<Segment>&,
                                       const CellPerm&, u128, int, int, unsigned) {}
#endif // MEL_GPU

// Retourne true si TOUTES les cellules ont été entièrement scannées (plage épuisée).
static bool run_neighbor_mode(u128 lo, u128 hi, int level, int N, unsigned nt,
                               bool force_gpu, bool force_cpu, bool gpu_only, int gpu_blocks, int gpu_threads) {
    u128 span = hi - lo + (u128)1;
    // free_bits : nombre de bits nécessaires pour couvrir [0, span). Pour une plage "bits"
    // classique (span = 2^(bits-1)), free_bits == bits - 1 exactement comme avant.
    int free_bits = u128_bitlen(span - (u128)1);
    if (!segment_fits(free_bits, N)) {
        std::cerr << "Erreur : " << N << " bits ne tient pas dans cette plage (" << free_bits
                  << " bit(s) utile(s) au maximum)\n";
        exit(1);
    }
    auto segments = build_segments(level, free_bits, N);
    for (auto &s : segments)
        if (!segment_fits(free_bits, s.len)) {
            std::cerr << "Erreur : le segment '" << s.nom << "' ne tient pas dans la plage.\n";
            exit(1);
        }
    const char *noms[] = {"", "prefixe", "milieu", "suffixe", "prefixe+milieu+suffixe"};
    int k = free_bits - N;                  // bits du motif d'une cellule
    const u128 total_cells = (u128)1 << k;  // nombre de cellules par type de segment
    CellPerm perm;
    {   // clés de la permutation : différentes à chaque lancement (ordre de visite aléatoire)
        std::random_device rd;
        std::mt19937_64 g(((uint64_t)rd() << 32) | rd());
        perm.init(k, g);
    }
    const double total_d = (double)total_cells;

    std::vector<int> good_gpus;
    if (!force_cpu) {
        auto t_gpu = std::chrono::steady_clock::now();
        good_gpus = gpu_select_devices();
        if (!good_gpus.empty())
            std::cout << good_gpus.size() << " GPU CUDA detecte(s) et valide(s) par auto-test en "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gpu).count() << " s.\n";
    }
    if (force_gpu && good_gpus.empty())
        std::cerr << "Note : aucun GPU CUDA utilisable (binaire compilé sans nvcc, aucun GPU détecté, "
                     "ou auto-test GPU échoué) ; bascule CPU.\n";
    bool use_gpu = !good_gpus.empty() && !force_cpu;
    // MODE HYBRIDE : par défaut, quand des GPU sont utilisés, TOUS les coeurs CPU travaillent aussi (neighbor_worker)
    // en parallèle des threads pilotes de GPU ; tous puisent dans la même file de cellules (g_nb_next).
    unsigned n_cpu = (use_gpu && gpu_only) ? 0 : nt;
    bool hybrid = use_gpu && n_cpu > 0;
    // Threads de préparation des jobs côté hôte, par GPU : peu nombreux quand les coeurs sont déjà occupés par les workers CPU.
    unsigned prep_nt = hybrid ? std::max(1u, std::min(4u, nt / ((unsigned)good_gpus.size() * 2u))) : nt;

    std::cout << "Plage : " << to_hex(lo) << " -> " << to_hex(hi);
    if (use_gpu)
        std::cout << " | GPU : " << good_gpus.size() << " (grille "
                  << (gpu_blocks > 0 ? std::to_string(gpu_blocks) : std::string("auto")) << "x" << gpu_threads << ")";
    if (n_cpu > 0) std::cout << " | threads CPU : " << n_cpu;
    if (hybrid) std::cout << " | MODE HYBRIDE CPU+GPU";
    std::cout << " | mode : voisinage " << noms[level] << " (" << N << " bits)"
              << " | ordre : aléatoire sans doublon (" << fmt_keys(total_d) << " cellules"
              << ", ordre différent à chaque lancement)"
              << " | pas de reprise\n";

    // Précalcule la table de fenêtre (WinTable) CPU de chaque segment UNE SEULE FOIS (nécessaire dès que des workers CPU
    // tournent ; inutile en mode GPU seul : run_neighbor_gpu_blocking construit sa propre table, format device).
    // Le point d'incrément d'un segment ne dépend que de son start_bit, fixe pour toute la session.
    std::vector<WinTable> seg_wt;
    if (n_cpu > 0) {
        seg_wt.resize(segments.size());
        for (size_t i = 0; i < segments.size(); ++i) {
            Point incPt = scalar_mul_fixed(u128_to_U256((u128)1 << segments[i].start_bit));
            build_wintable(seg_wt[i], incPt);
        }
    }

    g_count = 0;
    g_count_gpu = 0;
    g_nb_next = 0;
    g_nb_abandoned = false;
    auto t_start = std::chrono::steady_clock::now();
    std::atomic<bool> fin{false};
    std::thread disp([&] {
        auto t0 = t_start, tl = t0;
        uint64_t last = 0, last_g = 0;
        int sleep_ms = std::max(50, (int)(g_display_interval * 1000));
        int step = 0; // 0 : ligne immédiate, 1 : première mesure à 2 s, ensuite : intervalle normal
        while (!fin) {
            int wait_ms = step == 0 ? 0 : (step == 1 ? std::min(2000, sleep_ms) : sleep_ms);
            if (wait_ms > 0) {
            // attente par petits pas : l'arrêt (Ctrl-C, clé trouvée) reste immédiat
            auto tw = std::chrono::steady_clock::now();
            while (!fin && std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count() * 1000 < wait_ms)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            ++step;
            if (fin) break;
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - t0).count();
            double dt = std::chrono::duration<double>(now - tl).count();
            uint64_t c = g_count.load();
            double inst = dt > 0 ? (c - last) / dt : 0;
            last = c; tl = now;
            std::string split;
            if (hybrid) {
                uint64_t cg = g_count_gpu.load();
                double ig = dt > 0 ? (cg - last_g) / dt : 0;
                last_g = cg;
                split = " (GPU " + fmt_keys(ig) + "/s, CPU " + fmt_keys(std::max(0.0, inst - ig)) + "/s)";
            }
            double done = std::min((double)g_nb_next.load(), total_d); // essais lancés (cellules)
            double avg = el > 0 ? done / el : 0;
            {
                std::lock_guard<std::mutex> lp(g_print_mtx);
                printf("\r\033[K%s clés | %s clés/s%s | essai %s/%s | %.4f%% | écoulé %s | ETA max %s%s",
                       fmt_keys((double)c).c_str(), fmt_keys(inst).c_str(), split.c_str(),
                       fmt_keys(done).c_str(), fmt_keys(total_d).c_str(), 100.0 * done / total_d,
                       fmt_time(el).c_str(), fmt_time(avg > 0 ? (total_d - done) / avg : 1e18).c_str(),
                       fmt_targets().c_str());
                fflush(stdout);
            }
        }
    });

    // Workers CPU lancés EN PREMIER (non bloquant), puis les GPU (run_neighbor_gpu_blocking attend la fin de tous les
    // threads GPU) ; enfin on rejoint les threads CPU. Tous partagent g_nb_next : aucune cellule scannée deux fois.
    std::vector<std::thread> th;
    th.reserve(n_cpu);
    for (unsigned t = 0; t < n_cpu; ++t)
        th.emplace_back(neighbor_worker, std::cref(perm), total_cells, lo, span, std::cref(segments), std::cref(seg_wt));
    if (use_gpu)
        run_neighbor_gpu_blocking(good_gpus, lo, span, segments, perm, total_cells, gpu_blocks, gpu_threads, prep_nt);
    for (auto &t : th) t.join();

    fin = true; disp.join();
    // Plage épuisée = toutes les cellules réservées ET aucun thread n'a abandonné un essai en cours.
    bool exhausted = (u128)g_nb_next.load() >= total_cells && !g_nb_abandoned.load();
    if (exhausted) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\r\033[K%s clés (%llu) | 100.0000%% | écoulé %s%s",
               fmt_keys((double)g_count.load()).c_str(), (unsigned long long)g_count.load(),
               fmt_time(el).c_str(), fmt_targets().c_str());
    }
    printf("\n");
    return exhausted;
}

static bool self_test() {
    bool ok = true;
    auto check = [&](bool c, const char *name) { if (!c) { fprintf(stderr, "AUTO-TEST ECHEC : %s\n", name); ok = false; } };
    U256 ex;
    parse_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5", ex);
    Point g2 = jac_to_aff(jac_mul(U256{{2,0,0,0}}, G));
    check(eq(g2.x, ex), "2G");
    parse_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", ex);
    check(eq(G.x, ex), "G.x");
    // hash160(G) connu (adresse du puzzle 1 : 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH)
    unsigned char wb[20] = {0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
    unsigned char comp[33], h[20];
    point_to_compressed(G, comp);
    hash160_bytes(comp, 33, h);
    check(memcmp(h, wb, 20) == 0, "hash160(G)");
    check(eq(pack20(wb), pack20(h)), "pack20");
    // scalar_mul_fixed vs jac_mul sur plusieurs valeurs
    std::mt19937_64 rng(424242);
    for (int i = 0; i < 8; i++) {
        u128 k = ((u128)rng() << 64) | rng();
        k &= (((u128)1 << 80) - 1); if (k == 0) k = 1;
        Point a = scalar_mul_fixed(u128_to_U256(k));
        Point b = jac_to_aff(jac_mul(u128_to_U256(k), G));
        if (!(eq(a.x,b.x) && eq(a.y,b.y))) { check(false, "scalar_mul_fixed"); break; }
    }
    // table de fenêtre (HALF) vs jac_mul
    build_wintable(g_wt_main, G);
    for (int i : {1, 2, 255, 256, 511, 512}) {
        Point j = jac_to_aff(jac_mul(U256{{(u64)i,0,0,0}}, G));
        if (!(eq(j.x, g_wt_main.gx[i]) && eq(j.y, g_wt_main.gy[i]))) { check(false, "table de fenetre"); break; }
    }
    // hash160 vectorisé vs hash160_bytes
    {
        U256 xs[VLANES], hs[VLANES]; u64 ps[VLANES]; int bad = 0;
        for (int i = 0; i < VLANES; i++) {
            u128 k = ((u128)rng() << 64) | rng(); if ((k & (((u128)1<<80)-1)) == 0) k = i + 1;
            k &= (((u128)1 << 80) - 1); if (k == 0) k = i + 1;
            Point p = jac_to_aff(jac_mul(u128_to_U256(k), G));
            xs[i] = p.x; ps[i] = p.y.d[0] & 1;
        }
        hash160_vec(xs, ps, hs);
        for (int i = 0; i < VLANES; i++) {
            unsigned char cc[33], hh[20];
            cc[0] = ps[i] ? 0x03 : 0x02;
            x_to_bytes(xs[i], cc + 1);
            hash160_bytes(cc, 33, hh);
            if (!eq(hs[i], pack20(hh))) bad++;
        }
        check(bad == 0, "hash160_vec vs hash160_bytes");
    }
    // filtre/ensemble de cibles + scan complet multi-cibles : clés 12345 et 9000 dans la fenêtre
    // [8192, 16383], clé 20000 volontairement hors fenêtre (ne doit PAS être signalée).
    {
        TargetSet ts;
        const int tk[3] = {12345, 9000, 20000};
        for (int i = 0; i < 3; i++) {
            Point pk = jac_to_aff(jac_mul(U256{{(u64)tk[i],0,0,0}}, G));
            unsigned char tgt[20];
            point_to_compressed(pk, comp);
            hash160_bytes(comp, 33, tgt);
            ts.add("test", tgt, 0);
        }
        check(ts.find(ts.h[1]) == 1 && ts.find(ZERO) == -1, "TargetSet::find");
        Scratch sc;
        std::atomic<uint64_t> tc{0}; // compteur local : ne pollue pas g_count
        std::vector<std::pair<int64_t,int>> hits;
        Point start = scalar_mul_fixed(u128_to_U256(8192));
        scan_batch(g_wt_main, start, 8192, sc, tc, ts, [](){ return false; },
                   [&](int64_t h, int ti) { hits.push_back({h, ti}); });
        std::sort(hits.begin(), hits.end());
        check(hits.size() == 2 && hits[0] == std::make_pair((int64_t)(9000 - 8192), 1)
                               && hits[1] == std::make_pair((int64_t)(12345 - 8192), 0),
              "scan_batch multi-cibles");
        check(tc.load() == 8192, "scan_batch compteur");
    }
    // points de départ en progression arithmétique (préparation des jobs GPU) vs scalar_mul_fixed,
    // avec plusieurs replis modulo nchunks ; puis conversion par lot (point à l'infini au milieu)
    {
        const int cbt = 3; const uint64_t nch = 64, strd = 0x9E3779B97F4A7C15ULL | 1ULL, first = 37, cnt = 150;
        const u128 lot = (u128)1 << 20;
        std::vector<Point> pts(cnt);
        build_progression_starts(lot, cbt, strd, nch, first, cnt, pts.data());
        bool good = true;
        for (uint64_t j = 0; j < cnt && good; j++) {
            u128 key = lot + ((u128)(((first + j) * strd) & (nch - 1)) << cbt);
            Point r = scalar_mul_fixed(u128_to_U256(key));
            good = eq(pts[j].x, r.x) && eq(pts[j].y, r.y);
        }
        check(good, "build_progression_starts");
        std::vector<Jac> J(5); std::vector<Point> A(5); std::vector<U256> scr;
        for (int i = 0; i < 5; i++) J[i] = scalar_mul_fixed_jac(u128_to_U256((u128)(i * 7919 + 3)));
        J[2] = Jac{ZERO, ZERO, ZERO, true};
        jac_batch_to_aff(J.data(), A.data(), 5, scr);
        bool g2 = A[2].inf;
        for (int i = 0; i < 5 && g2; i++) {
            if (i == 2) continue;
            Point r = scalar_mul_fixed(u128_to_U256((u128)(i * 7919 + 3)));
            g2 = eq(A[i].x, r.x) && eq(A[i].y, r.y);
        }
        check(g2, "jac_batch_to_aff");
    }
    return ok;
}

// ---- Adresses du Bitcoin Puzzle n° 1 à 80 (index 0 = puzzle 1) ----
// Plage de clés du puzzle P : [2^(P-1), 2^P - 1]
static const char *PUZZLE_ADDR[80] = {
    "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb",
    "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA", "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e",
    "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k", "1PitScNLyp2HCygzadCh7FveTnfmpPbfp8",
    "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC", "1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK",
    "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV", "1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe",
    "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu", "1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot",
    "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1", "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk",
    "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW", "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY",
    "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm", "1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE",
    "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w", "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum",
    "14oFNXucftsHiUMY8uctg6N487riuyXs4h", "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv",
    "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac", "1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7",
    "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP", "1JVnST957hGztonaWK6FougdtjxzHzRMMg",
    "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k", "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY",
    "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT", "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps",
    "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE", "1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR",
    "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu", "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q",
    "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb", "1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1",
    "14iXhn8bGajVWegZHJ18vJLHhntcpL4dex", "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2",
    "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8", "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv",
    "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E", "1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N",
    "1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1", "1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF",
    "1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk", "1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP",
    "1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z", "1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv",
    "12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF", "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk",
    "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS", "15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim",
    "15K1YKJMiJ4fpesTVUcByoz334rHmknxmT", "1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy",
    "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa", "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi",
    "15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz", "1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf",
    "1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt", "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su",
    "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB", "1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi",
    "1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4", "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN",
    "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe", "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so",
    "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9", "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ",
    "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR",
    "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR",
    "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv",
    "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt", "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF",
    "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE", "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb",
    "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8", "1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe",
};

// Analyse "-pA" ou "-pApB" (A, B = numéros de puzzle). Format strict : -p<chiffres>[p<chiffres>].
static bool parse_puzzle_arg(const std::string &a, int &pa, int &pb) {
    size_t i = 2, j = i;
    while (j < a.size() && isdigit((unsigned char)a[j])) ++j;
    if (j == i || j - i > 3) return false;
    pa = atoi(a.substr(i, j - i).c_str());
    if (j == a.size()) { pb = pa; return true; }
    if (a[j] != 'p') return false;
    size_t k = ++j;
    while (j < a.size() && isdigit((unsigned char)a[j])) ++j;
    if (j != a.size() || j == k || j - k > 3) return false;
    pb = atoi(a.substr(k).c_str());
    return true;
}
static void print_usage(const char *prog) {
    std::cerr << "Usage : " << prog << " -pP | -pApB | <adresse> <bits 1-80 | début:fin en hex> [--seq] [--chunk N] "
              << "[-vN | -vvN | -vvvN | -vvvvN] [-t N] [-s T] [--gpu | --gpu-only | --cpu] [--gpu-blocks N]\n"
              << "  -pP             : puzzle numéro P (1-80) : adresse + plage [2^(P-1), 2^P-1] automatiques\n"
              << "  -pApB           : tous les puzzles de A à B (ex: -p66p70) cherchés EN MÊME TEMPS sur\n"
              << "                    l'intervalle [2^(A-1), 2^B-1] (le plus petit au plus grand puzzle)\n"
              << "                    plusieurs -p peuvent être combinés : -p66 -p70p72\n"
              << "  <bits 1-80>     : plage [2^(bits-1), 2^bits-1]\n"
              << "  <début:fin>     : plage personnalisée en hexadécimal (ex: 8000000000000000:ffffffffffffffff)\n"
              << "  -t N : nombre de cœurs CPU à utiliser (par défaut : tous), y compris en mode hybride CPU+GPU\n"
              << "  -s T : rafraîchit l'affichage toutes les T secondes (par défaut : 30)\n"
              << "  --gpu / --cpu   : force l'utilisation (ou non) du/des GPU CUDA (binaire compilé avec nvcc).\n"
              << "                    Par défaut : TOUS les GPU détectés ET TOUS les cœurs CPU travaillent ensemble\n"
              << "                    (mode hybride, y compris en mode -v) ; --cpu : CPU seul.\n"
              << "  --gpu-only      : GPU seuls (les cœurs CPU ne scannent pas).\n"
              << "  --gpu-blocks N  : nombre de blocs CUDA par lancement (défaut : automatique pour chaque carte,\n"
              << "                    selon son nombre de SM et l'occupation du kernel). Threads par bloc = GPU_BT, fixé à la\n"
              << "                    compilation (défaut 128, puissance de 2 ; --gpu-threads est ignoré).\n";
}
// Message de fin quand toutes les cibles n'ont pas été trouvées.
static void print_final_summary(bool exhausted, bool lead_nl) {
    int n = (int)g_ts.size(), left = g_remaining.load(), got = n - left;
    if (left == 0) return; // chaque clé a déjà été affichée par save_key
    if (lead_nl) std::cout << "\n";
    const char *why = exhausted ? "Plage entièrement scannée" : "Interrompu";
    if (n == 1) std::cout << why << (exhausted ? " : clé non trouvée.\n" : " (aucune clé trouvée).\n");
    else        std::cout << why << " : " << got << "/" << n << " clé(s) trouvée(s).\n";
}

int main(int argc, char **argv) {
    // ---- 1. Extraction des arguments -pP / -pApB (cumulables) ; le reste est traité plus bas ----
    std::vector<std::string> args(argv + 1, argv + argc), rest;
    bool puzzle_sel[MAX_TARGETS + 1] = {false};
    bool puzzle_mode = false;
    for (const std::string &a : args) {
        if (a.size() >= 3 && a[0] == '-' && a[1] == 'p' && isdigit((unsigned char)a[2])) {
            int pa = 0, pb = 0;
            if (!parse_puzzle_arg(a, pa, pb) || pa < 1 || pa > 80 || pb < 1 || pb > 80) {
                std::cerr << "Argument puzzle invalide : " << a
                          << " (attendu -pP ou -pApB avec 1 <= P <= 80, ex: -p66 ou -p66p70)\n";
                return 1;
            }
            if (pa > pb) std::swap(pa, pb);
            for (int p = pa; p <= pb; ++p) puzzle_sel[p] = true;
            puzzle_mode = true;
        } else rest.push_back(a);
    }

    // ---- 2. Cibles et intervalle ----
    u128 g_lo = 0, g_hi = 0;
    bool custom_range = false;
    int bits = 0;
    size_t opt_start = 0; // indice, dans 'rest', du premier argument optionnel
    if (puzzle_mode) {
        int pmin = 0, pmax = 0;
        for (int p = 1; p <= 80; ++p) {
            if (!puzzle_sel[p]) continue;
            unsigned char h20[20];
            if (!decode_address(PUZZLE_ADDR[p - 1], h20)) {
                std::cerr << "Erreur interne : adresse du puzzle #" << p << " invalide (checksum)\n";
                return 1;
            }
            g_ts.add(PUZZLE_ADDR[p - 1], h20, p);
            if (!pmin) pmin = p;
            pmax = p;
        }
        // Intervalle commun : borne inférieure du plus petit puzzle -> borne supérieure du plus grand.
        bits = pmax;
        g_lo = (u128)1 << (pmin - 1);
        g_hi = ((u128)1 << pmax) - 1;
        if (g_ts.size() == 1) {
            std::cout << "Puzzle #" << pmin << " | adresse : " << g_ts.addr[0] << "\n";
        } else {
            std::cout << "Puzzles #" << pmin << " à #" << pmax << " (" << g_ts.size() << " cibles) | intervalle commun : "
                      << to_hex(g_lo) << " -> " << to_hex(g_hi) << "\n";
            for (size_t i = 0; i < g_ts.size(); ++i)
                std::cout << "  #" << g_ts.puzzle[i] << " : " << g_ts.addr[i] << "\n";
        }
    } else {
        if (rest.size() < 2) { print_usage(argv[0]); return 1; }
        const std::string &addr = rest[0], &arg2 = rest[1];
        opt_start = 2;
        custom_range = arg2.find(':') != std::string::npos;
        if (custom_range) {
            if (!parse_hex_range(arg2, g_lo, g_hi)) {
                std::cerr << "Plage invalide : " << arg2 << " (attendu début:fin en hexadécimal, "
                          << "ex: 8000000000000000:ffffffffffffffff)\n";
                return 1;
            }
            if (g_lo == g_hi) {
                std::cerr << "La plage début:fin doit couvrir au moins 2 clés (début < fin)\n";
                return 1;
            }
            bits = u128_bitlen(g_hi);
        } else {
            bits = atoi(arg2.c_str());
        }
        if (bits < 1 || bits > 80) {
            std::cerr << (custom_range ? "La plage début:fin doit tenir sur 80 bits au maximum\n"
                                        : "bits doit être entre 1 et 80\n");
            return 1;
        }
        unsigned char h20[20];
        if (!decode_address(addr, h20)) { std::cerr << "Adresse P2PKH invalide (checksum)\n"; return 1; }
        g_ts.add(addr, h20, 0);
        if (!custom_range) { g_lo = (u128)1 << (bits - 1); g_hi = ((u128)1 << bits) - 1; }
    }
    g_remaining = (int)g_ts.size();
    for (int i = 0; i < MAX_TARGETS; ++i) g_tfound[i] = false;

    // ---- 3. Options ----
    int cb = 10;
    bool seq = false;
    int neighbor_level = 0, neighbor_bits = 0;
    unsigned nt_override = 0;
    bool force_gpu = false, force_cpu = false, gpu_only = false;
    int gpu_blocks = 0, gpu_threads = 0; // gpu_blocks 0 = automatique (par carte) ; gpu_threads : voir plus bas
    for (size_t i = opt_start; i < rest.size(); ++i) {
        const std::string &a = rest[i];
        if (a == "--seq") { seq = true; }
        else if (a == "--chunk" && i + 1 < rest.size()) { cb = atoi(rest[++i].c_str()); }
        else if (a == "-t" && i + 1 < rest.size()) {
            int v = atoi(rest[++i].c_str());
            if (v < 1) { std::cerr << "-t doit être suivi d'un nombre de cœurs >= 1\n"; return 1; }
            nt_override = (unsigned)v;
        }
        else if (a == "-s" && i + 1 < rest.size()) {
            double v = atof(rest[++i].c_str());
            if (v <= 0) { std::cerr << "-s doit être suivi d'une durée en secondes > 0\n"; return 1; }
            g_display_interval = v;
        }
        else if (a.size() >= 2 && a[0] == '-' && a[1] == 'v') {
            size_t p = 2; while (p < a.size() && a[p] == 'v') ++p;
            int level = (int)(p - 1);
            if (level < 1 || level > 4 || p >= a.size() || !isdigit((unsigned char)a[p])) {
                std::cerr << "Argument invalide : " << a << " (attendu -vN, -vvN, -vvvN ou -vvvvN)\n";
                return 1;
            }
            neighbor_level = level; neighbor_bits = atoi(a.c_str() + p);
        }
        else if (a == "--gpu") { force_gpu = true; }
        else if (a == "--cpu") { force_cpu = true; }
        else if (a == "--gpu-only") { gpu_only = true; force_gpu = true; }
        else if (a == "--gpu-blocks" && i + 1 < rest.size()) {
            int v = atoi(rest[++i].c_str());
            if (v < 1) { std::cerr << "--gpu-blocks doit être suivi d'un entier >= 1\n"; return 1; }
            gpu_blocks = v;
        }
        else if (a == "--gpu-threads" && i + 1 < rest.size()) {
            int v = atoi(rest[++i].c_str());
            if (v < 1 || v > 1024) { std::cerr << "--gpu-threads doit être suivi d'un entier entre 1 et 1024\n"; return 1; }
            gpu_threads = v;
        }
        else std::cerr << "Avertissement : argument ignoré : " << a << "\n";
    }
    cb = std::max(0, std::min(cb, bits - 1));
    if (gpu_threads > 0 && gpu_threads != GPU_BT)
        std::cerr << "Note : --gpu-threads " << gpu_threads << " ignore : la taille de bloc (arbre d'inversion en memoire "
                     "partagee) est fixee a la compilation, GPU_BT=" << GPU_BT << " (recompiler avec -DGPU_BT=N, puissance de 2).\n";
    gpu_threads = GPU_BT;

    U256 gx, gy;
    parse_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", gx);
    parse_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", gy);
    G.x = gx; G.y = gy; G.inf = false;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    unsigned nt = nt_override ? nt_override : std::max(1u, std::thread::hardware_concurrency());

    auto t_init = std::chrono::steady_clock::now();
    init_gtable(G);
    build_wintable(g_wt_main, G);
    if (!self_test()) { std::cerr << "Erreur : auto-test echoue, arret.\n"; return 1; }
    double dt_init = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_init).count();
    std::cout << "Tables precalculees (base fixe + fenetre " << HALF << ") et auto-tests OK en "
              << dt_init << " s | SIMD hash160 : " << VLANES << " voies\n";

    // gpu_blocks == 0 : grille automatique, choisie séparément pour chaque GPU (voir gpu_auto_blocks).

    if (neighbor_level > 0) {
        if (seq || cb != 10)
            std::cerr << "Note : --seq et --chunk sont ignorés en mode voisinage.\n";
        bool exhausted = run_neighbor_mode(g_lo, g_hi, neighbor_level, neighbor_bits, nt,
                                            force_gpu, force_cpu, gpu_only, gpu_blocks, gpu_threads);
        print_final_summary(exhausted, false);
        return 0;
    }

    u128 lo = g_lo;
    u128 span = g_hi - g_lo + (u128)1; // nombre réel de clés à couvrir (pas forcément une puissance de deux)
    {
        // Correctif (bug présent depuis la v23) : le nombre de chunks est stocké dans un uint64_t.
        // Au-delà de 2^62 chunks (span/2^cb), il débordait silencieusement (ex: puzzle 80 avec
        // cb=10 -> 2^69 chunks tronqué à 0, donc 1 seul chunk scanné puis « plage entièrement
        // scannée » à tort). On agrandit alors la taille de chunk pour rester sous 2^62 chunks.
        int span_bits = u128_bitlen(span - (u128)1);
        if (span_bits - cb > 62) {
            cb = span_bits - 62;
            std::cerr << "Note : intervalle très large, taille de chunk portée à 2^" << cb << ".\n";
        }
        u128 chunk_size = (u128)1 << cb;
        uint64_t nchunks_real = (uint64_t)((span + chunk_size - (u128)1) / chunk_size);
        g_nchunks = next_pow2_u64(nchunks_real == 0 ? 1 : nchunks_real);
    }

    std::random_device rd;
    uint64_t stride = seq ? 1 : ((((uint64_t)rd() << 32) | rd()) | 1);

    std::vector<int> good_gpus; // GPU détectés ET ayant réussi leur auto-test (vide si --cpu)
    if (!force_cpu) {
        auto t_gpu = std::chrono::steady_clock::now();
        good_gpus = gpu_select_devices();
        if (!good_gpus.empty())
            std::cout << good_gpus.size() << " GPU CUDA detecte(s) et valide(s) par auto-test en "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gpu).count() << " s.\n";
    }
    if (force_gpu && good_gpus.empty())
        std::cerr << "Note : aucun GPU CUDA utilisable (binaire compilé sans nvcc, aucun GPU détecté, "
                     "ou auto-test GPU échoué) ; bascule CPU.\n";
    bool use_gpu = !good_gpus.empty() && !force_cpu;
    // MODE HYBRIDE : tous les coeurs CPU (worker) ET tous les GPU (gpu_worker) puisent en même temps dans la même file de
    // chunks. Indices de g_cur : 0..n_cpu-1 = workers CPU, n_cpu..n_cpu+ngpu-1 = threads pilotes de GPU.
    const unsigned ngpu = use_gpu ? (unsigned)good_gpus.size() : 0;
    const unsigned n_cpu = (use_gpu && gpu_only) ? 0 : nt;
    const bool hybrid = use_gpu && n_cpu > 0;
    // Threads de préparation des jobs par GPU : peu nombreux quand les coeurs sont déjà occupés par les workers CPU.
    const unsigned prep_nt = hybrid ? std::max(1u, std::min(4u, nt / (ngpu * 2u))) : nt;

    g_cur.assign(n_cpu + ngpu, IDLE);

    std::cout << "Plage : " << to_hex(lo) << " -> " << to_hex(g_hi)
              << " | cibles : " << g_ts.size();
    if (use_gpu)
        std::cout << " | GPU : " << ngpu << " (grille "
                  << (gpu_blocks > 0 ? std::to_string(gpu_blocks) : std::string("auto")) << "x" << gpu_threads << ")";
    if (n_cpu > 0) std::cout << " | threads CPU : " << n_cpu;
    if (hybrid) std::cout << " | MODE HYBRIDE CPU+GPU";
    std::cout << " | mode : " << (seq ? "séquentiel" : "aléatoire sans doublon")
              << " | chunk : 2^" << cb << "\n";

    g_count = 0; // l'auto-test ne doit rien laisser dans le compteur
    g_count_gpu = 0;
    std::vector<std::thread> th;
    for (unsigned i = 0; i < ngpu; ++i) // GPU d'abord : leur première réservation de plage part sans attendre
        th.emplace_back(gpu_worker, n_cpu + i, good_gpus[i], lo, span, cb, stride, gpu_blocks, gpu_threads, prep_nt);
    for (unsigned i = 0; i < n_cpu; ++i) th.emplace_back(worker, i, lo, span, cb, stride);

    auto t0 = std::chrono::steady_clock::now(), tl = t0;
    uint64_t last = 0, last_g = 0;
    const double total = (double)span;
    auto all_idle = [&] {
        std::lock_guard<std::mutex> l(g_mtx);
        if (g_next < g_nchunks) return false;
        for (auto c : g_cur) if (c != IDLE) return false;
        return true;
    };

    int sleep_ms = std::max(50, (int)(g_display_interval * 1000));
    // On attend par petits pas de 50 ms (et non d'un seul bloc de 'sleep_ms', qui peut valoir
    // jusqu'à 30 s ou plus avec -s) afin qu'un Ctrl-C soit pris en compte quasi immédiatement :
    // sans ça, g_stop passait bien à true tout de suite (les threads s'arrêtaient vite), mais le
    // thread principal restait endormi jusqu'à la fin de l'intervalle d'affichage avant de s'en
    // apercevoir, joindre les threads et rendre la main — ce qui donnait l'impression d'un
    // programme qui ne répond pas au Ctrl-C.
    const int poll_ms = 50;
    int since_print_ms = sleep_ms; // force un premier affichage dès le premier passage
    while (!g_stop && !all_idle()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        since_print_ms += poll_ms;
        if (g_stop || all_idle()) break;
        if (since_print_ms < sleep_ms) continue;
        since_print_ms = 0;
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        double dt = std::chrono::duration<double>(now - tl).count();
        uint64_t c = g_count.load();
        double inst = (c - last) / dt, avg = c / el;
        last = c; tl = now;
        std::string split;
        if (hybrid) {
            uint64_t cg = g_count_gpu.load();
            double ig = dt > 0 ? (cg - last_g) / dt : 0;
            last_g = cg;
            split = " (GPU " + fmt_keys(ig) + "/s, CPU " + fmt_keys(std::max(0.0, inst - ig)) + "/s)";
        }
        double remaining = total - (double)c;
        {
            std::lock_guard<std::mutex> lp(g_print_mtx);
            printf("\r\033[K%s clés | %s clés/s%s | %.4f%% | écoulé %s | ETA max %s%s",
                   fmt_keys((double)c).c_str(), fmt_keys(inst).c_str(), split.c_str(), 100.0 * (total - remaining) / total,
                   fmt_time(el).c_str(), fmt_time(avg > 0 ? remaining / avg : 1e18).c_str(),
                   fmt_targets().c_str());
            fflush(stdout);
        }
    }
    g_stop = true;
    for (auto &t : th) t.join();

    bool exhausted = safe_index() >= g_nchunks && !g_gpu_lost.load(); // un GPU perdu = chunks possiblement non scannés
    if (exhausted) {
        // Dernière ligne de progression : sans elle, l'affichage resterait figé sur la dernière
        // mesure périodique (ex: 92 %) alors que la plage est terminée.
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> lp(g_print_mtx);
        printf("\r\033[K%s clés (%llu) | 100.0000%% | écoulé %s%s",
               fmt_keys((double)g_count.load()).c_str(), (unsigned long long)g_count.load(),
               fmt_time(el).c_str(), fmt_targets().c_str());
        fflush(stdout);
    }
    print_final_summary(exhausted, true);
}
