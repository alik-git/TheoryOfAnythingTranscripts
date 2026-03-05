---
layout: home
title: Theory of Anything Transcripts
nav_order: 1
---

# Theory of Anything Transcripts

> Unofficial AI-generated transcripts. These may contain mistakes.

This site publishes searchable transcript pages generated from the repository pipeline.

## Links To The Podcast

[Spotify](https://open.spotify.com/show/0bmymUJs50SVO1WkqfrccB) / [Apple Podcasts](https://podcasts.apple.com/us/podcast/the-theory-of-anything/id1503194218)

## Episodes

{% assign episode_pages = site.pages | where_exp: "p", "p.url contains '/episodes/' and p.name != 'index.md'" | sort: "nav_order" | reverse %}
{% if episode_pages.size > 0 %}
{% for p in episode_pages %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endfor %}
{% else %}
- No episode pages published yet.
{% endif %}

- [Full Episodes Section](./episodes/)

## Search

Use the built-in search box in the top-right to search all indexed pages.
