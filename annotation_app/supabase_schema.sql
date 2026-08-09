-- Private shared backend for the fingerprint subtype annotation app.
-- Run this once in the Supabase SQL editor before uploading the review package.

create table if not exists public.review_items (
    review_id text primary key,
    record_key text not null unique,
    image_path text not null unique,
    current_main_type text not null
        check (current_main_type in ('arch', 'whorl')),
    source_primary_code text not null
        check (source_primary_code in ('AU', 'WU')),
    recorded_pattern_codes text not null,
    alternative_type_warning boolean not null default false,
    permitted_subtypes text[] not null,
    created_at timestamptz not null default now(),
    check (cardinality(permitted_subtypes) > 0)
);

create table if not exists public.annotations (
    review_id text primary key
        references public.review_items(review_id) on delete restrict,
    confirmed_subtype text not null,
    confidence text not null
        check (confidence in ('high', 'medium', 'low')),
    review_action text not null
        check (review_action in ('accept', 'adjudicate', 'exclude')),
    main_type_issue text
        check (main_type_issue is null or main_type_issue in ('incorrect', 'uncertain')),
    review_notes text,
    reviewer_id text not null,
    first_reviewed_at timestamptz not null default now(),
    reviewed_at timestamptz not null default now(),
    revision integer not null default 1 check (revision > 0)
);

create table if not exists public.annotation_events (
    event_id bigint generated always as identity primary key,
    review_id text not null
        references public.review_items(review_id) on delete restrict,
    revision integer not null,
    reviewer_id text not null,
    annotation_snapshot jsonb not null,
    created_at timestamptz not null default now(),
    unique (review_id, revision)
);

create index if not exists annotations_reviewer_idx
    on public.annotations(reviewer_id);
create index if not exists annotations_action_idx
    on public.annotations(review_action);
create index if not exists annotation_events_review_idx
    on public.annotation_events(review_id, revision);

alter table public.review_items enable row level security;
alter table public.annotations enable row level security;
alter table public.annotation_events enable row level security;

-- The Streamlit server uses a service-role secret. Browser clients receive no
-- database or storage credentials and no anon/authenticated table access.
revoke all on public.review_items from anon, authenticated;
revoke all on public.annotations from anon, authenticated;
revoke all on public.annotation_events from anon, authenticated;

create or replace function public.save_subtype_annotation(
    p_review_id text,
    p_confirmed_subtype text,
    p_confidence text,
    p_review_action text,
    p_main_type_issue text,
    p_review_notes text,
    p_reviewer_id text,
    p_expected_revision integer
)
returns public.annotations
language plpgsql
security definer
set search_path = public
as $$
declare
    v_item public.review_items%rowtype;
    v_current_revision integer;
    v_saved public.annotations%rowtype;
begin
    select * into v_item
    from public.review_items
    where review_id = p_review_id;

    if not found then
        raise exception 'Unknown review ID: %', p_review_id;
    end if;

    if not (p_confirmed_subtype = any(v_item.permitted_subtypes)) then
        raise exception 'Subtype % is not permitted for %',
            p_confirmed_subtype, v_item.current_main_type;
    end if;
    if p_confidence not in ('high', 'medium', 'low') then
        raise exception 'Invalid confidence value';
    end if;
    if p_review_action not in ('accept', 'adjudicate', 'exclude') then
        raise exception 'Invalid review action';
    end if;
    if p_main_type_issue is not null
       and p_main_type_issue not in ('incorrect', 'uncertain') then
        raise exception 'Invalid main-type issue';
    end if;
    if nullif(btrim(p_reviewer_id), '') is null then
        raise exception 'Reviewer ID is required';
    end if;
    if p_confirmed_subtype = 'unclear' and p_review_action = 'accept' then
        raise exception 'An unclear subtype cannot be accepted';
    end if;
    if p_main_type_issue is not null and (
        p_confirmed_subtype <> 'unclear'
        or p_review_action <> 'adjudicate'
        or nullif(btrim(coalesce(p_review_notes, '')), '') is null
    ) then
        raise exception 'A main-type issue requires unclear, adjudicate, and notes';
    end if;
    if p_review_action in ('adjudicate', 'exclude')
       and nullif(btrim(coalesce(p_review_notes, '')), '') is null then
        raise exception 'Adjudication and exclusion require review notes';
    end if;

    select revision into v_current_revision
    from public.annotations
    where review_id = p_review_id
    for update;

    if not found then
        v_current_revision := 0;
    end if;

    if v_current_revision <> p_expected_revision then
        raise exception 'Revision conflict for %. Expected %, current %',
            p_review_id, p_expected_revision, v_current_revision
            using errcode = '40001';
    end if;

    insert into public.annotations (
        review_id,
        confirmed_subtype,
        confidence,
        review_action,
        main_type_issue,
        review_notes,
        reviewer_id,
        revision
    ) values (
        p_review_id,
        p_confirmed_subtype,
        p_confidence,
        p_review_action,
        p_main_type_issue,
        nullif(btrim(coalesce(p_review_notes, '')), ''),
        btrim(p_reviewer_id),
        v_current_revision + 1
    )
    on conflict (review_id) do update set
        confirmed_subtype = excluded.confirmed_subtype,
        confidence = excluded.confidence,
        review_action = excluded.review_action,
        main_type_issue = excluded.main_type_issue,
        review_notes = excluded.review_notes,
        reviewer_id = excluded.reviewer_id,
        reviewed_at = now(),
        revision = excluded.revision
    returning * into v_saved;

    insert into public.annotation_events (
        review_id, revision, reviewer_id, annotation_snapshot
    ) values (
        v_saved.review_id,
        v_saved.revision,
        v_saved.reviewer_id,
        to_jsonb(v_saved)
    );

    return v_saved;
end;
$$;

revoke all on function public.save_subtype_annotation(
    text, text, text, text, text, text, text, integer
) from public, anon, authenticated;
grant execute on function public.save_subtype_annotation(
    text, text, text, text, text, text, text, integer
) to service_role;
