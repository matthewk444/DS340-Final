"""Map Quick, Draw! country codes (ISO alpha-2) to 9 world regions.

Our own grouping, loosely based on UN subregions. Central Asia, the Caucasus
and the Balkans are grouped with Eastern Europe so that group has enough data.
"""

REGIONS = [
    "North America",
    "Latin America",
    "Western Europe",
    "Eastern Europe",
    "Middle East & Africa",
    "South Asia",
    "East Asia",
    "Southeast Asia",
    "Oceania",
]

_CODES = {
    "North America": "US CA BM PM GL",
    "Latin America": (
        "MX GT BZ SV HN NI CR PA CU JM HT DO PR TT BS BB LC VC GD AG DM KN AW CW "
        "SX KY VI VG TC AI MS MF BL GP MQ BR AR CL CO PE VE EC BO PY UY GY SR GF FK"
    ),
    "Western Europe": (
        "GB IE FR DE NL BE LU CH AT LI MC ES PT IT GR MT CY AD SM VA "
        "SE NO DK FI IS FO AX JE GG IM GI"
    ),
    "Eastern Europe": (
        "RU UA BY PL CZ SK HU RO BG MD EE LV LT RS HR SI BA ME MK AL XK "
        "GE AM AZ KZ UZ TM TJ KG"
    ),
    "Middle East & Africa": (
        "TR IL PS SA AE QA KW BH OM YE JO LB SY IQ IR EG MA DZ TN LY SD EH "
        "ZA NG KE GH ET TZ UG ZW ZM MZ MW MG AO CM CI SN ML BF NE TD CF CD CG GA GQ "
        "ST BJ TG SL LR GN GW GM CV MR SO DJ ER SS RW BI KM SC MU NA BW LS SZ SH RE"
    ),
    "South Asia": "IN PK BD LK NP BT MV AF",
    "East Asia": "CN JP KR KP TW HK MO MN",
    "Southeast Asia": "TH VN PH ID MY SG MM KH LA BN TL",
    "Oceania": "AU NZ PG FJ NC VU SB WS TO PF GU FM MH MP PW NR KI TV CK NU WF AS PN NF",
}

COUNTRY_TO_REGION = {
    code: REGIONS.index(region)
    for region, codes in _CODES.items()
    for code in codes.split()
}


def region_of(country_code):
    """Region index for a country code, or None if the code is unmapped."""
    return COUNTRY_TO_REGION.get(country_code)
