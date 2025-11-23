import random
import argparse

WORD_SIZE = 16


def seed_from_xy(x: int, y: int, word_size: int = WORD_SIZE) -> int:
    """
    Convierte la pareja (x, y) en una semilla entera para el PRNG.
    Asumimos que x e y son máscaras de 'word_size' bits.
    """
    mask = (1 << word_size) - 1
    x &= mask
    y &= mask
    # Semilla = x concatenado con y
    return (x << word_size) | y


def generar_mascaras_txt_xy(
    x_seed: int,
    y_seed: int,
    n: int,
    filename: str,
    word_size: int = WORD_SIZE,
):
    """
    Genera n máscaras de 'word_size' bits a partir de la semilla (x_seed, y_seed)
    y las guarda en un archivo de texto, una por línea, en formato hex 0x1234.
    """
    # 1) semilla entera a partir de (x_seed, y_seed)
    seed = seed_from_xy(x_seed, y_seed, word_size)

    # 2) PRNG determinista
    rng = random.Random(seed)

    # 3) parámetros de máscara
    mask_val = (1 << word_size) - 1
    hex_digits = word_size // 4  # 16 bits → 4 dígitos hex

    # 4) escribir archivo
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(n):
            m = rng.getrandbits(word_size) & mask_val
            f.write(f"0x{m:0{hex_digits}x}\n")


def parse_int_auto(s: str) -> int:
    """
    Convierte una cadena a int permitiendo decimal o hex (0x1234).
    """
    return int(s, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Generar máscaras de 16 bits a partir de una semilla (x, y)."
    )
    parser.add_argument(
        "--x",
        type=parse_int_auto,
        required=True,
        help="Máscara x de la semilla (ej: 0x1234 o 4660).",
    )
    parser.add_argument(
        "--y",
        type=parse_int_auto,
        required=True,
        help="Máscara y de la semilla (ej: 0xABCD o 43981).",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        help="Número de máscaras a generar (por defecto: 100).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="mascaras.txt",
        help="Nombre del archivo de salida (por defecto: mascaras.txt).",
    )

    args = parser.parse_args()

    generar_mascaras_txt_xy(
        x_seed=args.x,
        y_seed=args.y,
        n=args.num,
        filename=args.output,
    )

    print(f"Generadas {args.num} máscaras en {args.output} "
          f"usando semilla x={hex(args.x)}, y={hex(args.y)}")


if __name__ == "__main__":
    main()

#Comando: python generar_mascaras.py --x mask1 --y mask2 -n num_mascaras -o nombre_archivo.txt
