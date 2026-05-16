import os
GOOGLE_STUDIO_AI_API_KEY=os.getenv("GOOGLE_API_KEY")
MODEL = "gemini-3.1-flash-lite-preview"

from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_STUDIO_AI_API_KEY)

import re
import time
import json
import hashlib
import requests
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

import fitz  # PyMuPDF
import pandas as pd


def _text_from_generate_content_response(response: Any) -> str:
    """google-genai returns ``GenerateContentResponse`` with ``.text``, not ``.generations``."""
    try:
        t = response.text
    except Exception:
        t = None
    if isinstance(t, str) and t.strip():
        return t

    gens = getattr(response, "generations", None)
    if gens:
        g0 = gens[0]
        gt = getattr(g0, "text", None)
        if isinstance(gt, str) and gt.strip():
            return gt

    cands = getattr(response, "candidates", None) or []
    if cands:
        content = getattr(cands[0], "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts:
            chunks: list[str] = []
            for p in parts:
                pt = getattr(p, "text", None)
                if pt:
                    chunks.append(str(pt))
            if chunks:
                return "".join(chunks)

    raise AttributeError(
        f"No text on {type(response).__name__}; try response.candidates / SDK upgrade."
    )


class DataCenterSpecsExtraction(BaseModel):
    data_center_name: Optional[str] = None
    address: Optional[str] = None
    colocation_space_sqm: Optional[float] = None
    building_materials: List[str] = Field(default_factory=list)
    maximum_power_load_kw: Optional[float] = None
    standby_power_total_kw: Optional[float] = None
    cooling_configuration: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

class DataCenterLocation(BaseModel):
    latitude: Optional[str | float] = None
    longitude: Optional[str | float] = None

class ThermalCapacity(BaseModel):
    material: Optional[str] = None
    matched_material: Optional[str] = None
    phase: Optional[str] = None
    specific_heat_kj_per_kg_k: Optional[float] = None

class MaterialThermalCapacityExtraction(BaseModel):
    thermal_capacities: List[ThermalCapacity] = Field(default_factory=list)

class ProcessPDF:
    def extract_text(self, pdf_path, max_pages=99):
        pdf_path = Path(pdf_path)
        pages = []
        with fitz.open(pdf_path) as doc:
            for page_idx in range(min(len(doc), max_pages)):
                page = doc[page_idx]
                text = page.get_text("text")
                if text and text.strip():
                    pages.append(f"\n--- PAGE {page_idx + 1} ---\n{text}")
        return "\n".join(pages).strip()

    def chunk_text(self, text, chunk_size=9000, overlap=800):
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = max(end - overlap, end)
        return chunks

class ResponseParser:
    def json_from_model_output(self, output, original=None):
        output = output.strip()
        try:
            return json.loads(output)
        except Exception:
            pass
        output = re.sub(r"^```(?:json)?", "", output, flags=re.IGNORECASE).strip()
        output = re.sub(r"```$", "", output).strip()
        try:
            return json.loads(output)
        except Exception:
            pass
        match = re.search(r"\{.*\}", output, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return original

    def choose_best_value(self, values):
        """
        Simple merge strategy across chunks:
        pick the longest non-empty value, usually the most complete address/name.
        """
        clean_values = []
        for value in values:
            if value is None:
                continue
            value = str(value).strip()
            if value and value.lower() not in {"null", "none", "n/a", "unknown"}:
                clean_values.append(value)
        if not clean_values:
            return None
        unique_values = list(dict.fromkeys(clean_values))
        return max(unique_values, key=len)

    def create_nominatim_params(self, parsed, query_params):
        cleaned = {}
        for key in query_params:
            value = parsed.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if not value or value.lower() in {"null", "none", "unknown", "n/a"}:
                continue
            cleaned[key] = value
        return cleaned

class PromptBuilder:
    def extract_data_center_specs(self, text):
        return f"""
    You are an information extraction system.

    Extract the following fields from the PDF text:

    - data_center_name
    - address
    - colocation_space_sqm
    - building_materials
    - maximum_power_load_kw
    - standby_power_total_kw
    - cooling_configuration
    - notes

    Return ONLY valid JSON.
    Do not include markdown.
    Do not explain your answer.
    Do not guess. Use null when a field is not found.

    Rules:
    - data_center_name should be the name of the data center, facility, site, campus, or colocation location.
    - Address should be the full physical address if available.
    - If multiple addresses exist, choose the one most directly associated with the data center.
    - Preserve the address as a single string.
    - Return colocation space in square meters.
    - Normalize materials to simple distinct names when possible.
    - Only set maximum_power_load_kw if the document explicitly states a maximum or total power load in kW.
    - If the document lists standby/generator power ratings, sum them into standby_power_total_kw.
    - Keep cooling_configuration close to the document wording.
    - Do not invent values.

    Expected JSON schema:
    {{
     "data_center_name": string|null,
     "address": string|null,
     "colocation_space_sqm": number|null,
     "building_materials": [string],
     "maximum_power_load_kw": number|null,
     "standby_power_total_kw": number|null,
     "cooling_configuration": string|null,
     "notes": [string]
    }}

    PDF text:
    \"\"\"
    {text}
    \"\"\"
    """.strip()

    def nominatim_param(self, extracted_record):
        return f"""
    You are preparing structured search parameters for the Nominatim geocoding API.

    Given this extracted data center record, return ONLY valid JSON using this schema:

    {{
      "amenity": string or null,
      "street": string or null,
      "city": string or null,
      "county": string or null,
      "state": string or null,
      "country": string or null,
      "postalcode": string or null,
      "countrycodes": string or null
    }}

    Rules:
    - Use structured search parameters only.
    - Do NOT return q.
    - Do NOT invent values.
    - Use null when a component is unknown.
    - street should be only building number plus street name, for example "151 Front Street West".
    - city should be the city/locality.
    - state should be the province/state/region.
    - country should be the country name.
    - postalcode should be only the postal code.
    - countrycodes should be ISO 3166-1 alpha-2 lowercase, for example "ca" for Canada.
    - amenity may contain the data center or facility name, but do not include floor numbers or marketing symbols.
    - Remove symbols like ® unless they are needed for the place name.
    - If the address contains floor numbers, suite numbers, building notes, or facility specs, do not put those in street.

    Extracted record:
    {json.dumps(extracted_record, ensure_ascii=False, indent=2)}
    """.strip()

    def extract_material_thermal_capacity(self, text, materials):
        return f"""
    You are an information extraction system.

    Extract the isobaric mass heat capacity Cp for each requested material from the PDF text.

    Requested materials:
    {json.dumps(materials, ensure_ascii=False)}

    Return ONLY valid JSON.
    Do not include markdown.
    Do not explain your answer.
    Do not guess. Use null when a material is not found.

    Rules:
    - Return one result per requested material, in the same order.
    - Preserve the requested material name in the material field.
    - Use the table column named isobaric mass heat capacity, Cp.
    - The table unit J g^-1 K^-1 is numerically equal to kJ kg^-1 K^-1.
    - If the exact material is not listed, use the closest unambiguous substance name only when it is clearly the same material.
    - Put the table substance name in matched_material.
    - Put the table phase in phase when available.
    - Do not invent values.

    Expected JSON schema:
    {{
      "thermal_capacities": [
        {{
          "material": string,
          "matched_material": string|null,
          "phase": string|null,
          "specific_heat_kj_per_kg_k": number|null
        }}
      ]
    }}

    PDF text:
    \"\"\"
    {text}
    \"\"\"
    """.strip()


class LLMFlow:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def model_inference(self, prompt, response_schema):
        response = self.client.models.generate_content(
          model=self.model_name,
          contents=[prompt],
          config=types.GenerateContentConfig(
              response_mime_type="application/json",
              response_schema=response_schema,
            ),
        )
        return _text_from_generate_content_response(response)

    def extract_fields_from_text(self, prompt, response_schema):
        return self.model_inference(prompt, response_schema)

    def extract_params_from_address_line(self, prompt, response_schema=None):
        return self.model_inference(prompt, response_schema)

    def create_nominatim_params0(self, parsed, query_params, max_new_tokens=256):

        # print(decoded)
        # parsed = json.loads(decoded)
        cleaned = {}
        for key in query_params:
            value = parsed.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if not value or value.lower() in {"null", "none", "unknown", "n/a"}:
                continue
            cleaned[key] = value
        return cleaned

class LocationGrounding:
    NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"

    NOMINATIM_USER_AGENT = "DataCenterPDFGeocoder/0.1 (katerinaZakharov2000@gmail.com)"

    NOMINATIM_CACHE_PATH = Path(r"\nominatim_cache.json")

    ALLOWED_NOMINATIM_QUERY_PARAMS = {
        "amenity",
        "street",
        "city",
        "county",
        "state",
        "country",
        "postalcode",
        "countrycodes",
    }
    def load_nominatim_cache(self):
        if self.NOMINATIM_CACHE_PATH.exists():
            with open(self.NOMINATIM_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}


    def save_nominatim_cache(self, cache):
        with open(self.NOMINATIM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def make_cache_key(self, params):
        canonical = json.dumps(params, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def nominatim_search_structured(
        self,
        structured_params,
        limit=3,
        addressdetails=1,
        extratags=1,
        namedetails=1,
        cache=None,
        min_delay_seconds=2.1,
    ):
        # if cache is None:
            # cache = self.load_nominatim_cache()
        params = dict(structured_params)
        params.update({
            "format": "jsonv2",
        #     "limit": int(limit),
        #     "addressdetails": int(addressdetails),
        #     "extratags": int(extratags),
        #     "namedetails": int(namedetails),
        })
        params.pop("q", None)
        print(params)
        cache_key = self.make_cache_key(params)
        # if cache_key in cache:
        #     return cache[cache_key]
        headers = {
            "User-Agent": self.NOMINATIM_USER_AGENT,
            "Accept": "application/json",
        }
        time.sleep(min_delay_seconds)
        response = requests.get(
            self.NOMINATIM_ENDPOINT,
            params=params,
            headers=headers,
            timeout=30,
        )
        print("RESP", response.text)
        response.raise_for_status()
        data = response.json()
        #cache[cache_key] = {"request_params": params, "results": data}
        # self.save_nominatim_cache(cache)
        #return cache[cache_key]
        return {"request_params": params, "results": data}

    def normalize_nominatim_result(self, result):
        return {
            "latitude": result.get("lat"),
            "longitude": result.get("lon"),
            "geocoded_display_name": result.get("display_name"),
            "geocoded_class": result.get("class"),
            "geocoded_type": result.get("type"),
            "geocoded_importance": result.get("importance"),
            "geocoded_osm_type": result.get("osm_type"),
            "geocoded_osm_id": result.get("osm_id"),
            "geocoded_place_id": result.get("place_id"),
            "geocoded_boundingbox": result.get("boundingbox"),
        }

    def geocode_extracted_record(self, structured_params, cache=None):
        # if cache is None:
        #     cache = self.load_nominatim_cache()
        geocode_output = {
            "nominatim_amenity": structured_params.get("amenity"),
            "nominatim_street": structured_params.get("street"),
            "nominatim_city": structured_params.get("city"),
            "nominatim_county": structured_params.get("county"),
            "nominatim_state": structured_params.get("state"),
            "nominatim_country": structured_params.get("country"),
            "nominatim_postalcode": structured_params.get("postalcode"),
            "nominatim_countrycodes": structured_params.get("countrycodes"),
            "geocode_status": "not_attempted",
        }
        if not structured_params:
            return None

        try:
            response_data = self.nominatim_search_structured(
                structured_params,
                limit=3,
                cache=cache,
            )
            results = response_data.get("results", [])
            geocode_output["geocode_status"] = "matched" if results else "no_match"
            geocode_output["nominatim_result_count"] = len(results)
            if results:
                geocode_output.update(self.normalize_nominatim_result(results[0]))

            return geocode_output

        except Exception as e:
            print(f"error2: {e}")
            import traceback
            traceback.print_exc()
            return None

PDF_processer = ProcessPDF()
prompt_builder = PromptBuilder()
llm_flow = LLMFlow(client, MODEL)
response_parser = ResponseParser()
location_grounding = LocationGrounding()

class DataCenter:
    def __init__(self):
        self.data_center_specs = DataCenterSpecsExtraction().model_dump()
        self.data_center_location = DataCenterLocation().model_dump()
        self.location_cache = None#location_grounding.load_nominatim_cache()

    def get_specs(self, pdf_path: str):
        text = PDF_processer.extract_text(pdf_path)
        if not text: return self.data_center_specs
        chunks = PDF_processer.chunk_text(text)
        chunk_results = []
        for chunk in chunks:
            prompt = prompt_builder.extract_data_center_specs(chunk)
            extracted = llm_flow.extract_fields_from_text(prompt, DataCenterSpecsExtraction)
            parsed = response_parser.json_from_model_output(extracted, self.data_center_specs)
            chunk_results.append(parsed)
        for key in self.data_center_specs:
            self.data_center_specs[key] = response_parser.choose_best_value(
                [item.get(key) for item in chunk_results]
            )
        return self.data_center_specs

    def get_location(self, address):
        prompt = prompt_builder.nominatim_param(address)
        decoded = llm_flow.extract_params_from_address_line(prompt)
        parsed = response_parser.json_from_model_output(decoded)
        params = response_parser.create_nominatim_params(parsed, location_grounding.ALLOWED_NOMINATIM_QUERY_PARAMS)
        # TODO add checks for postal codes: for CA should be 6symbols (+space)
        params.pop('postalcode', None)
        # return self.data_center_location
        return location_grounding.geocode_extracted_record(params, self.location_cache)

class Material:
    MATERIAL_HEAT_CAPACITY_TABLE = [
        {"matched_material": "Air (sea level, dry, 0 C)", "phase": "gas", "specific_heat_kj_per_kg_k": 1.0035},
        {"matched_material": "Air (typical room conditions)", "phase": "gas", "specific_heat_kj_per_kg_k": 1.012},
        {"matched_material": "Aluminum", "phase": "solid", "specific_heat_kj_per_kg_k": 0.897},
        {"matched_material": "Ammonia", "phase": "liquid", "specific_heat_kj_per_kg_k": 4.700},
        {"matched_material": "Animal tissue", "phase": "mixed", "specific_heat_kj_per_kg_k": 3.500},
        {"matched_material": "Antimony", "phase": "solid", "specific_heat_kj_per_kg_k": 0.207},
        {"matched_material": "Argon", "phase": "gas", "specific_heat_kj_per_kg_k": 0.5203},
        {"matched_material": "Arsenic", "phase": "solid", "specific_heat_kj_per_kg_k": 0.328},
        {"matched_material": "Beryllium", "phase": "solid", "specific_heat_kj_per_kg_k": 1.820},
        {"matched_material": "Bismuth", "phase": "solid", "specific_heat_kj_per_kg_k": 0.123},
        {"matched_material": "Cadmium", "phase": "solid", "specific_heat_kj_per_kg_k": 0.231},
        {"matched_material": "Carbon dioxide", "phase": "gas", "specific_heat_kj_per_kg_k": 0.839},
        {"matched_material": "Chromium", "phase": "solid", "specific_heat_kj_per_kg_k": 0.449},
        {"matched_material": "Copper", "phase": "solid", "specific_heat_kj_per_kg_k": 0.385},
        {"matched_material": "Diamond", "phase": "solid", "specific_heat_kj_per_kg_k": 0.509},
        {"matched_material": "Ethanol", "phase": "liquid", "specific_heat_kj_per_kg_k": 2.440},
        {"matched_material": "Gasoline (octane)", "phase": "liquid", "specific_heat_kj_per_kg_k": 2.220},
        {"matched_material": "Glass", "phase": "solid", "specific_heat_kj_per_kg_k": 0.840},
        {"matched_material": "Gold", "phase": "solid", "specific_heat_kj_per_kg_k": 0.129},
        {"matched_material": "Granite", "phase": "solid", "specific_heat_kj_per_kg_k": 0.790},
        {"matched_material": "Graphite", "phase": "solid", "specific_heat_kj_per_kg_k": 0.710},
        {"matched_material": "Helium", "phase": "gas", "specific_heat_kj_per_kg_k": 5.193},
        {"matched_material": "Hydrogen", "phase": "gas", "specific_heat_kj_per_kg_k": 14.300},
        {"matched_material": "Hydrogen sulfide", "phase": "gas", "specific_heat_kj_per_kg_k": 1.015},
        {"matched_material": "Iron", "phase": "solid", "specific_heat_kj_per_kg_k": 0.449},
        {"matched_material": "Lead", "phase": "solid", "specific_heat_kj_per_kg_k": 0.129},
        {"matched_material": "Lithium", "phase": "solid", "specific_heat_kj_per_kg_k": 3.580},
        {"matched_material": "Lithium at 181 C", "phase": "solid(?)", "specific_heat_kj_per_kg_k": 4.233},
        {"matched_material": "Lithium at 181 C", "phase": "liquid", "specific_heat_kj_per_kg_k": 4.379},
        {"matched_material": "Magnesium", "phase": "solid", "specific_heat_kj_per_kg_k": 1.020},
        {"matched_material": "Mercury", "phase": "liquid", "specific_heat_kj_per_kg_k": 0.1395},
        {"matched_material": "Methane at 2 C", "phase": "gas", "specific_heat_kj_per_kg_k": 2.191},
        {"matched_material": "Methanol", "phase": "liquid", "specific_heat_kj_per_kg_k": 2.140},
        {"matched_material": "Molten salt", "phase": "liquid", "specific_heat_kj_per_kg_k": 1.560},
        {"matched_material": "Nitrogen", "phase": "gas", "specific_heat_kj_per_kg_k": 1.040},
        {"matched_material": "Neon", "phase": "gas", "specific_heat_kj_per_kg_k": 1.030},
        {"matched_material": "Oxygen", "phase": "gas", "specific_heat_kj_per_kg_k": 0.918},
        {"matched_material": "Paraffin wax", "phase": "solid", "specific_heat_kj_per_kg_k": 2.500},
        {"matched_material": "Polyethylene", "phase": "solid", "specific_heat_kj_per_kg_k": 2.302},
        {"matched_material": "Silica (fused)", "phase": "solid", "specific_heat_kj_per_kg_k": 0.703},
        {"matched_material": "Silver", "phase": "solid", "specific_heat_kj_per_kg_k": 0.233},
        {"matched_material": "Sodium", "phase": "solid", "specific_heat_kj_per_kg_k": 1.230},
        {"matched_material": "Steel", "phase": "solid", "specific_heat_kj_per_kg_k": 0.466},
        {"matched_material": "Tin", "phase": "solid", "specific_heat_kj_per_kg_k": 0.227},
        {"matched_material": "Titanium", "phase": "solid", "specific_heat_kj_per_kg_k": 0.523},
        {"matched_material": "Tungsten", "phase": "solid", "specific_heat_kj_per_kg_k": 0.134},
        {"matched_material": "Uranium", "phase": "solid", "specific_heat_kj_per_kg_k": 0.116},
        {"matched_material": "Water at 100 C (steam)", "phase": "gas", "specific_heat_kj_per_kg_k": 2.030},
        {"matched_material": "Water at 25 C", "phase": "liquid", "specific_heat_kj_per_kg_k": 4.181},
        {"matched_material": "Water at 100 C", "phase": "liquid", "specific_heat_kj_per_kg_k": 4.216},
        {"matched_material": "Water at -10 C (ice)", "phase": "solid", "specific_heat_kj_per_kg_k": 2.050},
        {"matched_material": "Zinc", "phase": "solid", "specific_heat_kj_per_kg_k": 0.387},
        {"matched_material": "Asphalt", "phase": "solid", "specific_heat_kj_per_kg_k": 0.920},
        {"matched_material": "Brick", "phase": "solid", "specific_heat_kj_per_kg_k": 0.840},
        {"matched_material": "Concrete", "phase": "solid", "specific_heat_kj_per_kg_k": 0.880},
        {"matched_material": "Glass, silica", "phase": "liquid", "specific_heat_kj_per_kg_k": 0.840},
        {"matched_material": "Glass, crown", "phase": "liquid", "specific_heat_kj_per_kg_k": 0.670},
        {"matched_material": "Glass, flint", "phase": "liquid", "specific_heat_kj_per_kg_k": 0.503},
        {"matched_material": "Glass, borosilicate", "phase": "liquid", "specific_heat_kj_per_kg_k": 0.753},
        {"matched_material": "Gypsum", "phase": "solid", "specific_heat_kj_per_kg_k": 1.090},
        {"matched_material": "Marble, mica", "phase": "solid", "specific_heat_kj_per_kg_k": 0.880},
        {"matched_material": "Sand", "phase": "solid", "specific_heat_kj_per_kg_k": 0.835},
        {"matched_material": "Soil", "phase": "solid", "specific_heat_kj_per_kg_k": 0.800},
        {"matched_material": "Water", "phase": "liquid", "specific_heat_kj_per_kg_k": 4.181},
        {"matched_material": "Wood", "phase": "solid", "specific_heat_kj_per_kg_k": 1.700},
    ]

    MATERIAL_ALIASES = {
        "reinforced concrete": "concrete",
        "structural concrete": "concrete",
        "cement": "concrete",
        "structural steel": "steel",
        "steel frame": "steel",
        "steel framing": "steel",
        "brickwork": "brick",
        "masonry brick": "brick",
        "silica glass": "glass silica",
        "crown glass": "glass crown",
        "flint glass": "glass flint",
        "borosilicate glass": "glass borosilicate",
        "ice": "water at 10 c ice",
        "steam": "water at 100 c steam",
    }

    def __init__(self, model=None, tokenizer=None, info_table_path="/context/table_of_specific_heat_capacities.pdf"):
        self.model = model
        self.tokenizer = tokenizer

        self.material_thermal_capacity = MaterialThermalCapacityExtraction().model_dump()

        self.info_table_path = info_table_path
        self.pdf_processor = globals().get("PDF_processer", ProcessPDF())
        self.prompt_builder = globals().get("prompt_builder", PromptBuilder())
        self.response_parser = globals().get("response_parser", ResponseParser())
        self.llm_flow = LLMFlow(model, tokenizer) if model is not None and tokenizer is not None else globals().get("llm_flow")

        self.material_reference_table = pd.DataFrame(self.MATERIAL_HEAT_CAPACITY_TABLE)
        self.material_reference_lookup = {
            self.normalize_material_name(row["matched_material"]): row
            for row in self.MATERIAL_HEAT_CAPACITY_TABLE
        }

    def normalize_material_name(self, material):
        if material is None:
            return ""
        material = str(material).lower()
        material = material.replace("(", " ").replace(")", " ")
        material = re.sub(r"[^a-z0-9]+", " ", material)
        return re.sub(r"\s+", " ", material).strip()

    def default_thermal_capacity_results(self, materials):
        return [
            ThermalCapacity(material=material).model_dump()
            for material in materials
        ]

    def parse_specific_heat_value(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"[-+]?\d*\.?\d+", str(value).replace(",", ""))
        if not match:
            return None
        return float(match.group(0))

    def records_from_parsed_response(self, parsed):
        if isinstance(parsed, dict):
            if isinstance(parsed.get("thermal_capacities"), list):
                return parsed.get("thermal_capacities")
            if isinstance(parsed.get("materials"), list):
                return parsed.get("materials")
            if "material" in parsed:
                return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []

    def merge_thermal_capacity_results(self, materials, chunk_results):
        results = self.default_thermal_capacity_results(materials)
        results_by_material = {
            self.normalize_material_name(item["material"]): item
            for item in results
        }
        for parsed in chunk_results:
            for record in self.records_from_parsed_response(parsed):
                if not isinstance(record, dict):
                    continue
                material_key = self.normalize_material_name(record.get("material"))
                if material_key not in results_by_material:
                    continue
                result = results_by_material[material_key]
                value = self.parse_specific_heat_value(record.get("specific_heat_kj_per_kg_k"))
                if value is not None:
                    result["specific_heat_kj_per_kg_k"] = value
                if record.get("matched_material"):
                    result["matched_material"] = record.get("matched_material")
                if record.get("phase"):
                    result["phase"] = record.get("phase")
        return results

    def lookup_material_reference(self, material):
        material_key = self.normalize_material_name(material)
        alias_key = self.MATERIAL_ALIASES.get(material_key, material_key)

        if alias_key in self.material_reference_lookup:
            return self.material_reference_lookup[alias_key]

        for reference_key, reference_record in self.material_reference_lookup.items():
            if alias_key and (alias_key == reference_key or alias_key in reference_key or reference_key in alias_key):
                return reference_record
        return None

    def add_reference_values(self, material_results):
        for result in material_results:
            if result.get("specific_heat_kj_per_kg_k") is not None:
                continue
            reference = self.lookup_material_reference(result.get("material"))
            if reference is None:
                continue
            result["matched_material"] = reference["matched_material"]
            result["phase"] = reference["phase"]
            result["specific_heat_kj_per_kg_k"] = reference["specific_heat_kj_per_kg_k"]
        return material_results

    def get_thermal_capacity(self, materials):
        if not materials:
            return []

        text = ""
        if Path(self.info_table_path).exists():
            text = self.pdf_processor.extract_text(self.info_table_path)

        chunk_results = []
        if text and self.llm_flow is not None:
            chunks = self.pdf_processor.chunk_text(text)
            for chunk in chunks:
                prompt = self.prompt_builder.extract_material_thermal_capacity(chunk, materials)
                extracted = self.llm_flow.extract_fields_from_text(
                    prompt, MaterialThermalCapacityExtraction
                )
                parsed = self.response_parser.json_from_model_output(extracted, self.material_thermal_capacity)
                chunk_results.append(parsed)

        material_results = self.merge_thermal_capacity_results(materials, chunk_results)
        return self.add_reference_values(material_results)


def extract_pdf_resources_bundle(
    pdf_path: Union[str, Path],
    *,
    try_geocode: bool = True,
) -> Dict[str, Any]:
    """
    HTTP/API wrapper: runs the existing ``DataCenter`` + ``Material`` pipeline and
    returns a JSON-serializable dict. On failure (e.g. LLM or IO), returns
    ``ok: False`` with ``error`` — no alternate extraction path.
    """
    path = Path(pdf_path)

    if not path.is_file():
        return {
            "ok": False,
            "error": f"PDF not found: {path}",
            "specs": {},
            "thermal_capacities": [],
            "location": None,
            "warnings": [],
        }
    try:
        dc = DataCenter()
        specs = dc.get_specs(str(path))
        materials = specs.get("building_materials") or []
        if isinstance(materials, str):
            try:
                materials = json.loads(materials)
            except Exception:
                materials = [materials]
        if not isinstance(materials, list):
            materials = []
        material_model = Material()
        thermal = material_model.get_thermal_capacity(materials)
        location = None
        if try_geocode and specs.get("address"):
            location = dc.get_location({"address": specs["address"]})
        
        return {
            "ok": True,
            "specs": specs,
            "thermal_capacities": thermal,
            "location": location,
            "warnings": [],
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "specs": {},
            "thermal_capacities": [],
            "location": None,
            "warnings": [str(exc)],
        }


if __name__ == "__main__":
    single_pdf = "/content/ibx_tr1_en.pdf"
    DC = DataCenter()
    dc_specs = DC.get_specs(single_pdf)
    print(dc_specs)
    dc_location = DC.get_location({"address": dc_specs["address"]})

    M = Material()
    thermal_capacity_list = M.get_thermal_capacity(eval(dc_specs["building_materials"]))
    print(thermal_capacity_list)