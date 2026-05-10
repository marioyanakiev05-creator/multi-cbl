## Share project with the trainer. Not clean but saves time and space
using Pkg
Pkg.activate(".")

## Necessary imports

# Fast iteration, cleaner stack traces... and duh
using Revise, RelevanceStacktrace, ProgressMeter

using DuckDB, DataFrames

# Handle Python packages and call them (RDKit)
using CondaPkg, PythonCall 

const Chem = pyimport("rdkit.Chem")

## Helper values
const PARQUET_PATHS = ["src/parquet-files/data/IR_data_chunk00$(i)_of_009.parquet" for i in 1:9]
const DB_PATH = "cache.duckdb" # Can use ":memory:" to use in-memory caching

# Need to validate that this is what we really need -efe 
# outdated
const FG_SMARTS = Dict(
  "Chlorine" => "[Cl]",
  "Ester Linkage" => "[CX3](=O)[OX2][CX4]",
  "Aromatic Ring" => "c1ccccc1",
  "Methyl Branch" => "[CH3][CX4H]([CH2])",
  "Ethylene Backbone" => "[CH2][CH2]"
)

## DuckDB init
con = DBInterface.connect(DuckDB.DB, DB_PATH) # IDK WHY BLUE SQUIGGLY LINE
DuckDB.query(con, """
    PRAGMA memory_limit='6GB';
    PRAGMA threads=12;
""")
## Process Parquet files
const table_name = "IRSpectra"

parquet_paths_sql = join(map(x -> "'" * x * "'", PARQUET_PATHS), ", ") # magic

# convert to native duckdb format dont do this if u dont know what
# this does (make your data pipelines faster maybe eventually) 
# dont run it it will probably make you bluescreen if ur on windows and hold you hostage for an hour
# if ur on something else but i wont delete this because i spent way too long writing it
# ok actually i tried another implementation and believe me or not this is the best way to do it so go ahead and run it
idk_what_this_returns = DuckDB.query(con, """
    PRAGMA memory_limit='6GB';
    PRAGMA threads=12;
    CREATE TABLE $table_name AS
    SELECT smiles, ir_spectra
    FROM read_parquet([$parquet_paths_sql]);
""")
## drop a column if u need to
#DuckDB.execute(con, """
#    ALTER TABLE IRSpectra DROP COLUMN "'Frequency(cm^-1)'";
#""")
# pro tip: if you have the same value for all rows its not inefficient and
# if u dropped a column like that it was all deduplicated and the space u gained is like kilobytes max
##
DuckDB.query(con, """
    VACUUM;
""")
##
DuckDB.query(con, """
    DESCRIBE IRSpectra;
""")
##
#DuckDB.query(con, """
#    ALTER TABLE IRSpectra
#    ALTER COLUMN ir_spectra SET DATA TYPE FLOAT[12000];
#""")
