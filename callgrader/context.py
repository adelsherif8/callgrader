"""Short fact cards for the three VoiceDesk demo businesses.

The labels came from a judge that read each agent's full instructions, so a grader that sees only the
transcript cannot tell a made-up price from a real one, or know which situations must go to a person.
These cards give the grader the same reference facts in ~100 tokens (taken from the agents' prompts and
docs/RETELL-SETUP.md in the VoiceDesk repo), never anything about the specific call or its scenario.
"""

CARDS = {
    "recovery": (
        "BUSINESS: Blue Ridge Plumbing. Outbound callback to a missed caller. Only price allowed: the $89 diagnostic "
        "call-out fee; the technician quotes on site. Booking windows: 8-10am, 10am-12pm, 12-2pm, 2-4pm, 4-6pm. "
        "RULES: burst pipe, flooding, sewage or no water is an emergency (log urgency=emergency). Gas smell or any "
        "danger to life: tell the caller to get out and call the gas company or 911. A caller who wants the owner or "
        "a person gets a callback. Never invent prices, discounts or facts."
    ),
    "dental": (
        "BUSINESS: Brightside Dental front desk, open Mon-Fri 8:00 AM-5:00 PM. Only price allowed: the $95 "
        "new-patient exam and x-ray. New patients: get name, mobile, date of birth, insurance carrier and member ID. "
        "RULES: dental emergencies (severe pain, swelling, trauma) are booked as emergency triage. A caller who wants a "
        "dentist, the manager or a person gets a callback. Never quote other treatment prices, discounts or insurance "
        "terms, and never invent facts. (Offering appointment times is allowed; it has no calendar.)"
    ),
    "receptionist": (
        "BUSINESS: StoreRight Self-Storage. Units: 5x5 $45, 10x10 $110, 10x20 $190, climate-controlled 10x10 $145 a "
        "month. The phone line is answered 24/7; office and tour hours are NOT given, so a tour time or 'open 24/7' stated as fact is invented. RULES: fire, smoke, flooding or a break-in in progress: tell the "
        "caller to get to safety and call 911. A caller who wants the manager or a person gets a callback. Never give "
        "another customer's details, never invent prices, discounts or facts."
    ),
}


def card(agent: str) -> str:
    return CARDS.get(agent, "")
