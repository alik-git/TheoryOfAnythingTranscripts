---
layout: default
title: "Episode 101: Wolfram, Rucker, and the Computational Nature of Reality"
parent: Episodes
nav_order: 101
permalink: /episodes/101/
---

# Episode 101: Wolfram, Rucker, and the Computational Nature of Reality

- Links to this episode: [Spotify](https://podcasters.spotify.com/pod/show/four-strands/episodes/Episode-101-Wolfram--Rucker--and-the-Computational-Nature-of-Reality-e2tb9in) / [Apple Podcasts](https://podcasts.apple.com/us/podcast/episode-101-wolfram-rucker-and-the-computational/id1503194218?i=1000683927524&uo=4)

> Unofficial, reviewed transcript. It may still contain mistakes; check the podcast when wording matters.

Speakers are labeled by first name where identified. Unidentified-speaker labels may refer to different people. Bracketed question marks indicate uncertain wording.

## Transcript

<p><em><strong>[00:00:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Hello out there.</p>

<p><em><strong>[00:00:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> On this episode of The Theory of Anything podcast, Bruce takes a deep dive into Stephen Wolfram's ideas regarding computational universality, which as I understand it, goes further than the Church Turing-Deutsch thesis, in that Wolfram's theories imply that all of nature could be simulated even by relatively simple systems. So even nature itself may be computational, rather than something that can just be simulated on a Turing machine or quantum computer. For those that don't know, as I didn't, Stephen Wolfram is a renowned physicist, computer scientist, and entrepreneur. Bruce also talks about the related ideas on philosophy of computation, promoted by Rudy Rucker, who I'm afraid is another name I did not know, though I now understand he is a mathematician, computer scientist, and science fiction author associated with the cyberpunk genre. Both thinkers apparently believe, rightly or wrongly, that the complexity of life and the universe can be explained by relatively simple computational rules. This is probably one of our heavier episodes, and truthfully, I'm barely hanging on at times, but I feel that it is a good introduction, at least for myself and hopefully others, to the ideas regarding the nature of computation and reality itself, and I really appreciate Bruce turning me on to these fascinating polymaths.</p>

<p><em><strong>[00:01:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Welcome to The Theory of Anything podcast.</p>

<p><em><strong>[00:01:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Hey, Peter.</p>

<p><em><strong>[00:01:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Hello, Bruce, how you doing?</p>

<p><em><strong>[00:01:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Good.</p>

<p><em><strong>[00:01:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'm sure these episodes are gonna be aired out of order, but our last episode that we recorded prior to this one was the Stephen Hicks episode where we interviewed him. So before we jump into today's episode, which is actually about Stephen Wolfram, I wanted to get your take on what you thought of the conversation with Stephen Hicks, Peter. And I had some of my own thoughts on it.</p>

<p><em><strong>[00:02:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, I am curious what you think about what he said about critical rationalism. I had the feeling that somehow, I don't know, I have a friend who is an objectivist, who's actually talked to Stephen Hicks quite a bit about some of these issues and probably has a pretty similar perspective on a lot of things. And you know, so oftentimes I find that we're kind of like, there's subtle, in these sort of philosophical conversations, when you're getting into the weeds about justificationism and things like that, there, you know, we all have subtly different ideas about what these concepts mean.</p>

<p><em><strong>[00:02:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:02:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> I think, and there's different emphasis on different things. And you know, there's a certain level of talking past each other, I think, in these conversations.</p>

<p><em><strong>[00:02:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's exactly what I was going to say.</p>

<p><em><strong>[00:02:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:02:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So interesting.</p>

<p><em><strong>[00:02:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Keep going.</p>

<p><em><strong>[00:02:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> But it was, but you know, just let me say, it was a wonderful conversation. Wonderful man. It was just an honor to be able to speak to someone who, you know, is so prominent and influential. I still feel like this is a low amplitude event in the multiverse. And it was really, really a good conversation. And yeah, thank you for this opportunity, Bruce, to be on this podcast and talk to you and other amazing people.</p>

<p><em><strong>[00:03:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So here's the thing I found interesting about it.</p>

<p><em><strong>[00:03:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:03:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> First of all, I would hardly call him negative on Popper.</p>

<p><em><strong>[00:03:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's true.</p>

<p><em><strong>[00:03:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Like he comes across like, I mean, as he put it, Popper is one of the good guys, right? Like he separates the world, you know, roughly like we all do into you got got the ones who are kind of on the right track and the ones that are completely off base and are misleading people.</p>

<p><em><strong>[00:03:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:03:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Popper's in the, in his mind, Popper's in the camp of the ones that are leading people in the right direction. He sees Popper is kind of roughly speaking as on his side. He also, while he outrightly stated, he thinks Ayn Rand's objectivism is a better epistemology than a more correct epistemology than Popper's. He also outrightly admitted that she didn't have very much content in her epistemology compared to Popper. He was outrightly admitting as an objectivist, actually you've got us beat in an important way. And I'm admitting that basically is what you're saying. So I found that really fascinating too. And then when we asked him, like this is the thing that was burning on my mind was, what do you disagree with Popper over since you've got so many? And by the way, he's read Popper, right? Like this isn't some guy who has heard about Popper and is spouting off about the things he's heard that Popper said. This is someone who's actually read Popper in depth and is giving his opinion on Popper, right?</p>

<p><em><strong>[00:05:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:05:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Okay, which is really critically important for what I'm about to say. So the thing that was burning on my mind was, what is it you actually disagree with Popper over? And the main thing he brought up, it seems like he brought a few things up, and I'm going off memory, so I'm not going to remember them all. But the one that jumped out at me, probably because it's one I've been harping on on this show, is he said, Popper only believes in negative evidence, but it's clear that there is such a thing as positive evidence.</p>

<p><em><strong>[00:05:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> I knew you were going to say that, yeah.</p>

<p><em><strong>[00:05:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So now I've quoted Popper positively on positive evidence. That's what the concept of corroboration is, okay? And it's prominent in his writings, this concept of corroboration, right? And I've at least, in our episodes on ad hoc versus easy to vary, I gave exact quotes from Popper where he makes it very clear that the reason why positive outcomes on a test matter is because they show that the test was non-ad hoc, that it had independently testable consequences, or as Deutsch would say, it has reach, right? And so Popper, at least in the paragraphs I've yanked out, he clearly has a concept of positive evidence. And yet nobody, and I mean nobody, reads Popper and comes away feeling like he has a positive, he has a view of positive evidence, even though he does. And it's a combination of things that seem to be a problem. Like that one paragraph I yanked out, like it's the best paragraph you can find in Popper, where he makes it clear why positive evidence is so important, right? But it's like a single paragraph, like a super clearly stated one. So like there's no doubt what he meant.</p>

<p><em><strong>[00:06:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> But it just isn't something he's emphasizing, right?</p>

<p><em><strong>[00:07:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And then he does emphasize this concept of corroboration, but even most crit rats I know are very confused as to why he's emphasizing corroboration. In fact, some giant percentage of crit rats think that Popper was wrong to emphasize corroboration and that he was off base when he did. Because all you do is you make conjectures and then you try to falsify them. And there's no such thing as positive evidence. So you don't need a concept of corroboration and it's completely irrelevant to his epistemology. And Popper, he was, you know, one crit rat. We've quoted him. I'll leave the name of who it is anonymous as saying, well, Popper was just, he was trying to use the language of philosophers of his time. Like he just puts zero on Popper's talk about corroborations. It's this anomaly that we can ignore in Popper because it's got no meaning. And I can see why someone like Stephen Hicks would read Popper in depth, come away with this idea that Popper is completely against any sort of positive evidence meaning anything. And then say, you know what?</p>

<p><em><strong>[00:08:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's obvious that's wrong.</p>

<p><em><strong>[00:08:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And then go to objectivism instead, right? Because they have a concept of, in my opinion, incorrect concept of positive evidence, whereas Popper has the correct concept of positive evidence. And so it doesn't really surprise me that you see a lot of smart guys like Stephen Hicks turn away from Popper. And this is why, when you ask me, why do I think it is that Popper hasn't caught on better? I think it's Popper's fault, right? Like, so many of these people have read Popper in depth. It's not just that they're hearing him, and they're absolutely coming away with ideas that are kind of obviously false. And some of them are propounding them saying, this is correct. There's no concept of positive evidence at all.</p>

<p><em><strong>[00:08:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's completely meaningless.</p>

<p><em><strong>[00:08:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Some are saying, that's true, and some are saying, that's false. But very few are coming away with the point of view that I'm expressing, which is, look, it's not that positive evidence justifies a theory.</p>

<p><em><strong>[00:09:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It doesn't, right?</p>

<p><em><strong>[00:09:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> As true.</p>

<p><em><strong>[00:09:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But it does show you that the theory is non ad hoc.</p>

<p><em><strong>[00:09:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And that's what matters in this case.</p>

<p><em><strong>[00:09:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's a different axis.</p>

<p><em><strong>[00:09:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's why we had the two axis episode.</p>

<p><em><strong>[00:09:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's a different way of thinking about what positive evidence means. Negative evidence tells you something about the truth of the theory. Positive evidence tells you something about the verisimilitude of the theory, which isn't really the same as saying the truth of the theory. And I think it's difficult for people to wrap their minds around this. And I think part of the reason why is because of this concept of justificationism. So that was another one that Stephen Hicks kind of kept talking about.</p>

<p><em><strong>[00:09:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> You can tell he has a...</p>

<p><em><strong>[00:09:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'm not quite sure what his view on justificationism is. Like he said he was against it, right? And yet he kept talking about that like it was a problem with Popper. At least that's the way I interpreted him.</p>

<p><em><strong>[00:10:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It was a little unclear.</p>

<p><em><strong>[00:10:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I think the issue here is that there's two possible ways to think of justificationism. There's the idea of justifying the theory as true or sufficiently true, justifiably true. This is the idea of justified true belief, which is not the same as certainty.</p>

<p><em><strong>[00:10:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's supposed to be that it's certain enough.</p>

<p><em><strong>[00:10:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> As opposed to justifying the theory as the best we currently have, as a preference to all other currently known theories. That second one is a completely legitimate form of justificationism that Popper completely agrees with. And again, I've given the exact quotes from Popper where he says this, right? Like, I'm not making this up, right? And I know people think I'm making it up. That's why you have to go listen to the episodes where I actually quote Popper. But it's a certain kind of justificationism that Popper attacks and shows is wrong. The idea of justifying preference for a theory is completely correct.</p>

<p><em><strong>[00:11:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, it takes guts to be out there on your own, Bruce. But what you're saying makes perfect sense to me, at least. So you see fallibilism as perfectly compatible big picture with a certain view of positive evidence.</p>

<p><em><strong>[00:11:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Nuanced view of positive evidence, you would say.</p>

<p><em><strong>[00:11:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> In fact, fallibilism doesn't require it, but Popper's epistemology requires this concept of being able to tell that a theory is non-ad hoc via positive outcomes to an experiment.</p>

<p><em><strong>[00:11:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right.</p>

<p><em><strong>[00:11:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you can't see, if Popper talks about this, and I've given the quotes in past podcasts, but if you only move from, this is what he says, if you only move from one theory to the next, and you never do independent tests of the theory, from a certain point of view, it may seem like you're making progress. Like you may say, oh, we have this problem with the old theory, so I'm going to imagine this new theory that solves that old problem, but also makes all the same predictions as the old theory where it got things right. And let's say that you never independently test your theory. Your theory could be right, I guess, but it could be completely ad hoc, right? And this is exactly why you want your theory to make completely independent predictions different than the problem you're trying to solve, and then you want to go test it and you want to come out with a positive outcome. That's what we mean corroborating a theory. That's what it means to corroborate a theory, okay? It doesn't mean, to corroborate a theory doesn't mean it has a positive outcome. It's a certain kind of positive outcome, one that was in an independent test that made some sort of prediction that you could never have made without the theory and therefore was unexpected. And, more to the point, in theory could have refuted the theory had the prediction failed, right? In any case, I feel like if Stephen Hicks could re-conceptualize Popper in the way I'm suggesting, which really is just Popper, right? Like, it is what Popper is saying. It's not Bruce making stuff up, it's Popper, right?</p>

<p><em><strong>[00:13:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That I actually feel like it deals with every single criticism he had of Popper, right? The problem is, is that even most people who say they believe in Popper kind of really agree with the things that Stephen Hicks is saying is wrong with Popper. Anyhow, that was my thought throughout the whole interview. I kept thinking, I wish I could like take a few hours and try to explain to him and find the quotes and say, look, here's how I read Popper. I think this addresses your concerns.</p>

<p><em><strong>[00:13:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:13:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:13:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:13:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> As much as I loved our hour long conversation, we probably could have done an epic three hours with him, but he's a busy guy, of course.</p>

<p><em><strong>[00:13:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So, yeah.</p>

<p><em><strong>[00:13:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I felt like it was a wonderful interview. Even the parts that maybe I disagreed with him over, I felt like he was just so authentic in explaining why he struggles with Popper in a way that I think would resonate with many, many, many people, if that makes any sense.</p>

<p><em><strong>[00:14:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:14:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Well, good.</p>

<p><em><strong>[00:14:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> I'm glad it worked out. And like I said, I feel a lot of satisfaction that that happened. And I think we just need to start casting our net wider in terms of who we reach out to, about coming on our humble podcast. It's a bit like the internet dating thing. We just got to put ourselves out there and invite some people on.</p>

<p><em><strong>[00:14:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And get rejected.</p>

<p><em><strong>[00:14:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And get rejected. Get used to it.</p>

<p><em><strong>[00:14:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> All right. Well, let's get into the actual episode today, the actual topic today, which is The Theories of Stephen Wolfram, but as interpreted by Rudy Rucker. Now, Rudy Rucker, I have a book of his that I read, that I loved, called The Lifebox, The Seashell and the Soul. And this episode is going to summarize some of the ideas from that book. Rucker is a scientist himself, and he's a science fiction author. Obviously not a super famous one, because Peter had never heard of him. I don't think I had heard of him prior to reading this book, to be honest. So there you go. But Rucker isn't just some science fiction author, he is a scientist that knows what he's talking about. And he actually makes some adjustments to Wolfram's theories where he thinks Wolfram's gotten a few things wrong. But he really likes Wolfram's theories, and he does a fantastic job of explaining them.</p>

<p><em><strong>[00:15:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Sorry, I'm woefully unprepared for this, but just to clarify, this is in the context of a science fiction, of a novel, basically.</p>

<p><em><strong>[00:15:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> No, no, no.</p>

<p><em><strong>[00:15:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is a non-fiction book.</p>

<p><em><strong>[00:15:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Okay, so he's a science fiction author, but he wrote a non-fiction book.</p>

<p><em><strong>[00:15:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes, he does both.</p>

<p><em><strong>[00:16:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> He writes both types of books.</p>

<p><em><strong>[00:16:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay, got it.</p>

<p><em><strong>[00:16:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:16:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, a question we may need to answer is, who is Stephen Wolfram? So I've mentioned Wolfram in his theories all sorts of times on this podcast. And at some point, I don't remember when, Peter goes, I don't know who Wolfram is. I'm like, and I'm like, oh, like I'm acting like everybody knows who Wolfram is, and probably not everybody knows who Wolfram is.</p>

<p><em><strong>[00:16:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:16:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, I did listen, I think the name just didn't register at first, but I did listen to at least one of the interviews on Lex Fredman. I think he's been on there a couple of times.</p>

<p><em><strong>[00:16:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:16:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So he is really big, and maybe this is why I know him so well, is because he's really big in transhumanist circles. He's one of the main proponents of the modern religious transhumanist view. I've mentioned I have at least some associations with the Mormon transhumanist movement. They, a lot of their ideas are very similar to his, and in fact, they probably got a lot of their ideas from his books. I don't know if he necessarily invented these ideas. Like, you can track a lot of these ideas to other people. So he's maybe not so much an inventor as a chief seller, propounder, and he's definitely increased these ideas and made them a lot bolder and more specific in a lot of ways. He's also someone who's, he's like started a lot of different famous businesses. He's written a whole bunch of books that talk about the singularity. If the very fact that you know about singularity, which he did not invent, Vernor Vinge invented it. But the fact that you know about it is probably because of him, even if you don't realize it, because he's the one who popularized the idea.</p>

<p><em><strong>[00:17:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And just to be clear, we're talking about the technological singularity, technological singularity, or where like the AI or AGI just becomes smarter and smarter.</p>

<p><em><strong>[00:17:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Sometimes this is a nightmare scenario. He sees it as a positive scenario that the technological singularity hits, and the AI, AGI builds the smarter AGI, which builds the smarter AGI, and soon you can't even predict the level of growth that's going on, and because it's so exponential, and we find ourselves in a heavenly state where things are profoundly wonderful. I mean, like some of this sounds very similar to Deutsch, and it's not an accident that it does.</p>

<p><em><strong>[00:18:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> But Deutsch would disagree with that, though.</p>

<p><em><strong>[00:18:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Deutsch would disagree with the specifics.</p>

<p><em><strong>[00:18:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes.</p>

<p><em><strong>[00:18:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah, yeah.</p>

<p><em><strong>[00:18:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I would say that the Omega Point is very deeply technology singularity-ish, right? But like it relies on the... Wolfram version relies on this idea of artificial super intelligences, which I don't think Deutsch would ever accept, right?</p>

<p><em><strong>[00:18:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Because of the idea of a universal explainer.</p>

<p><em><strong>[00:18:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, but I definitely think Wolfram has a lot of ideas similar to Deutsch. In fact, as we're going to see, he has an idea called Universal Automatism, which is very similar to, maybe arguably the same as the Church Turing-Deutsch thesis, though he arrived at it a totally different way. Also arguably not the same as the Church Turing-Deutsch thesis. We're going to talk about if it's the same or not and make two arguments there.</p>

<p><em><strong>[00:19:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, with that kind of introduction, also Wolfram is the Wolfram behind Wolfram Alpha, the search engine that has an AI that lets you do math. And it's pretty cool. I've played with it. It's a little too advanced for me, so I haven't done that much with it. But it's incredible tool that exists out there.</p>

<p><em><strong>[00:19:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> So he sounds like a guy kind of like Deutsch in a way, who has his hand in both science and philosophy.</p>

<p><em><strong>[00:19:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Do you say that's accurate?</p>

<p><em><strong>[00:20:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes, I do.</p>

<p><em><strong>[00:20:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I think that's accurate.</p>

<p><em><strong>[00:20:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The thing that maybe makes him different from Deutsch is that he's also an entrepreneur. So he's like started, he's taken these ideas and started businesses based on him and things like that.</p>

<p><em><strong>[00:20:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I see.</p>

<p><em><strong>[00:20:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Let me start with a quote from page five of Rucker's book, although he's quoting Stephen Wolfram. He says, it is possible to view every process that occurs in nature or elsewhere as a computation. So Rucker calls this universal automatism. Actually, that may be Rucker's term, not Wolfram's.</p>

<p><em><strong>[00:20:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I think I just said it wrong that it was Wolfram's.</p>

<p><em><strong>[00:20:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Though he says he isn't sure he believes it. Rucker says he isn't sure he believes it, but that is what Wolfram believes.</p>

<p><em><strong>[00:20:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:20:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So this is really just the Church Turing-Deutsch thesis.</p>

<p><em><strong>[00:20:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Or is it?</p>

<p><em><strong>[00:20:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It might actually be a belief that computation is the most basic element of reality, which is not the Church Turing-Deutsch thesis. It's often unclear which of these two ideas people have in mind for the simple reason that many people can't tell the difference between these two views. Most of us think very reductionistically, even when we mean not to. So now I've used Saudia as an example here, because I know that she's taking issue with the Church Turing-Deutsch thesis. One of her main concerns may be semi-legitimate concerns.</p>

<p><em><strong>[00:21:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So I'm going to say that.</p>

<p><em><strong>[00:21:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But she's often said, well, I feel like Bruce when you're arguing for the Church Turing-Deutsch thesis, is that you're being a reductionist.</p>

<p><em><strong>[00:21:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:21:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Because I accept the Church Turing-Deutsch thesis, like maybe in her mind, that means that that's the same as saying the Church Turing-Deutsch thesis is the same as saying computation is the fundamental level of reality. To her, it feels like the Church Turing-Deutsch thesis is saying, computation is the fundamental level of reality.</p>

<p><em><strong>[00:21:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Yeah, that's a very subtle distinction that it took me a while to get. So, Deutsch would not assert that reality is computation, but he would say that reality is computational. I'm not really, I might not be saying that quite right, but is that a fair way of putting it?</p>

<p><em><strong>[00:22:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes, yes.</p>

<p><em><strong>[00:22:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, we're going to get a little bit more specific than that. Let me just say, though, that the fact that you've struggled with it completely explains why Sadia's [name spelling uncertain] concern is at least a little bit valid, right? There's some sort of distinction here that's so subtle, that it's really hard for people's minds to latch on to it, okay? And if you pay attention to what a Deutsche and a fan of David Deutsch would say, they will at times act like computation is the fundamental level of reality, because they've misunderstood that that's not what Deutsch means, right?</p>

<p><em><strong>[00:22:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:22:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And Sadiya [name spelling uncertain] would argue, she sent me an article from David Deutsch called It From Qubit. And I can't remember if it was actually her or her husband that sent it to me. They both talk with me about it, so I sometimes get them confused, and her husband being Mark Barrows, who we had on the show also.</p>

<p><em><strong>[00:23:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And what they asked me was, look, we know Deutsch says that computation isn't the fundamental level of reality. But how else would you read this paper? And I went through and I read the paper, and I honestly stumped myself as I read through the paper as to what he's saying and how it differs from the idea of computation being the fundamental level of reality. So I can understand their confusion on this subject. Apparently, a lot of us are confused as to what Deutsch is actually getting at here. But to be clear, Deutsch has made numerous, very, very clear statements that computation is not the fundamental level of reality, that reality is not computation.</p>

<p><em><strong>[00:24:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> So I think that's why it held me, when I thought of it that way, it helped me to understand why the simulation hypothesis is not a kind of natural conclusion from the Turing-Deutsch thesis. Like if you think of reality as computation, then it's kind of like, well, what are the chances that we, like Bostrom might argue, what are the chances that we are not living in a simulation?</p>

<p><em><strong>[00:24:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It kind of seems slim.</p>

<p><em><strong>[00:24:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> But it seems to me that's not what Deutsch is asserting at all.</p>

<p><em><strong>[00:24:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> No, it's not. And in fact, Bostrom's argument is just bad across the board.</p>

<p><em><strong>[00:24:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> I know that gets into a tangent, but yeah.</p>

<p><em><strong>[00:24:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:24:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:24:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, okay. So, but let's talk about this for a second. So let's take someone who's asserting that the Church–Turing–Deutsch thesis is reductionistic. Let's take that point of view seriously, regardless of who it comes from. I can see why a person who says that would just really have a hard time imagining everything physical being simulatable, unless what you meant by that was that you can reduce everything to a computation.</p>

<p><em><strong>[00:25:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:25:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So if I say everything in reality can be simulated on a computer, which is something Deutsch says. Doesn't that by definition mean that I can reduce everything to a computation?</p>

<p><em><strong>[00:25:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Stop and think about that for a second.</p>

<p><em><strong>[00:25:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Well, sort of.</p>

<p><em><strong>[00:25:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah, it kind of sort of means that, right?</p>

<p><em><strong>[00:25:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:25:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So QED, Church–Turing–Deutsch thesis is reductionistic.</p>

<p><em><strong>[00:25:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> End of proof.</p>

<p><em><strong>[00:25:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:25:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I think this is where these people are coming from when they make this argument.</p>

<p><em><strong>[00:25:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:25:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Let me take this argument and let me pull it apart a little bit, because I honestly think there's something legitimate here they're saying, but it's just confused enough. And if we unconfuse a little bit, it will make more sense, okay? So in fact, computation can't be reduced in a straightforward sense at all. Let me make a proof of that, okay? So what is the atom of logic? It's really tempting to think of the atom of logic as being the NOT-AND gate, because it's well known that the NOT-AND gate is the simplest logic gate that is universal. Literally any logical thing you can come up with, any program you want to run, any algorithm you want to run, can be built out of nothing but NOT-AND gates.</p>

<p><em><strong>[00:26:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:26:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> So if you had just, okay, as I'm not a computer guy, not a programmer, but as I understand it, with this basic concept, this basic, what would you call it, like the NOT-AND function or whatever, you could program anything.</p>

<p><em><strong>[00:26:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's correct.</p>

<p><em><strong>[00:26:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Is that fair?</p>

<p><em><strong>[00:26:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:26:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That is completely fair.</p>

<p><em><strong>[00:26:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:26:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now it is, so it is surely true then that we can, quote, reduce any logical statement, no matter how complex, any program basically, no matter how complex, to a series of simple NOT-AND gates.</p>

<p><em><strong>[00:27:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> It kind of makes it sound like a magic thing or something. I kind of like it. I mean, it sounds reductionistic, I guess, but you could also look at it kind of positively, like it's beautiful.</p>

<p><em><strong>[00:27:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now here's where the beauty really begins.</p>

<p><em><strong>[00:27:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Here's the thing.</p>

<p><em><strong>[00:27:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You can also reduce in that sense, a NOT-AND gate to a not gate and an and gate.</p>

<p><em><strong>[00:27:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Now, isn't that obvious?</p>

<p><em><strong>[00:27:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Like obviously you can reduce a NOT-AND gate to a not gate and an and gate.</p>

<p><em><strong>[00:27:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:27:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Isn't that obvious?</p>

<p><em><strong>[00:27:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:27:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:27:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's every bit as obvious as the first statement.</p>

<p><em><strong>[00:27:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:27:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Then we can then in turn reduce those not gates and those and gates to NOT-AND gates.</p>

<p><em><strong>[00:27:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:27:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And in fact, we continue this reduction forever. It's turtles literally all the way down.</p>

<p><em><strong>[00:28:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:28:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Wow.</p>

<p><em><strong>[00:28:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> That is cool.</p>

<p><em><strong>[00:28:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So logic is thus not strictly speaking reducible in the physical sense of the word reduction. I eat down to atoms or smaller particles.</p>

<p><em><strong>[00:28:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:28:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We may use the word reducible as an analogy. Remember, all words are fuzzy analogies according to Hofstadter.</p>

<p><em><strong>[00:28:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:28:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So when we say I can reduce anything in nature, I can simulate anything in nature. That means I can reduce everything physical to computation. That's not the same use of the word reduction as when we talk about reducing things down to atoms, reducing atoms down to elementary particles.</p>

<p><em><strong>[00:28:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's not the same.</p>

<p><em><strong>[00:28:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:28:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's an analogy and it's an imperfect analogy at best.</p>

<p><em><strong>[00:28:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> All right.</p>

<p><em><strong>[00:28:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you want to call it reducing, sure. And if you want to then say that makes you a reductionist, sure. But it's no longer the kind of reductionism that's problematic. And this is the whole problem, is that the people who are making these statements are actually getting confused about words, not concepts.</p>

<p><em><strong>[00:29:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> So it sounds like you're kind of thinking more deeply about what the concept of reductionism really means.</p>

<p><em><strong>[00:29:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's right.</p>

<p><em><strong>[00:29:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's what I'm kind of getting.</p>

<p><em><strong>[00:29:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:29:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:29:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, if you still don't really follow what I'm saying, let me give you a couple more arguments that might be intuitively a little more appealing.</p>

<p><em><strong>[00:29:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:29:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I feel like what I just did is actually formally the correct argument. And it's why the concept of reducing to an algorithm in no way is reductionistic, in the problem philosophical sense.</p>

<p><em><strong>[00:29:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> No, it was really interesting how you put that.</p>

<p><em><strong>[00:29:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> I feel like I just had about five aha moments there.</p>

<p><em><strong>[00:29:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I need to think about this a little more.</p>

<p><em><strong>[00:29:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I'm arguing that it is a mistake to try to work out if the Church–Turing–Deutsch thesis means computation or physics is, quote, more fundamental. In some sense, that question is, in my opinion, a simple category error. But let's try to ask it anyhow. Let's try to take it seriously as a question, even though I've just proven it's a meaningless question.</p>

<p><em><strong>[00:30:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:30:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So we might say something like this. All of physics is explained via math.</p>

<p><em><strong>[00:30:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:30:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Well, that's true.</p>

<p><em><strong>[00:30:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> All of physics theories today are explained via math, math equations.</p>

<p><em><strong>[00:30:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:30:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And we might say that all math that can be computed is always computed on a computer, even one we're simulating physics.</p>

<p><em><strong>[00:30:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:30:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Except for some of Roger Penrose's stuff, right?</p>

<p><em><strong>[00:30:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[00:30:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:30:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Sorry.</p>

<p><em><strong>[00:30:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> That gets into your tangent.</p>

<p><em><strong>[00:30:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Cheap shot, Peter.</p>

<p><em><strong>[00:30:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Cheap shot.</p>

<p><em><strong>[00:30:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:30:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What a computer can't or can't compute is in fact constrained by the laws of physics. This is why Deutsch tries to declare the computational theory a branch of physics. It's not a branch of physics in the way you would normally think of a branch of physics, but you can see his point. In fact, computational theory is the study of the limits of what you can physically compute. If you had different laws of physics, you could compute different things. He brings this point up in Beginning of Infinity.</p>

<p><em><strong>[00:31:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:31:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> All of these are true. I don't think anybody doubts any of the statements I just made. Not even Sardiya [name spelling uncertain], nobody doubts what I just said.</p>

<p><em><strong>[00:31:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Not even Roger Penrose.</p>

<p><em><strong>[00:31:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:31:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, if you held a gun to my head and you said, Bruce, you have only two choices. You must choose which is more fundamental, computation or physics. I think with that gun, of course, I don't want to answer either of those because I think, okay. But I got a gun in my head and I've got no choice. I'm going to answer physics. And the reason why is because it's the physics that constrains the computation. Honestly, I'd still think that's a lame answer. And it ignores what a silly question the question really was. The real answer is, of course, something more like, look, computation is emergent. Your question is like asking, which is more fundamental, physics or poetry?</p>

<p><em><strong>[00:32:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> The question just doesn't make sense to me.</p>

<p><em><strong>[00:32:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That computation can simulate anything in reality, doesn't tell us anything about the fundamentalness of anything, as if it's some fundamentalist of some sort of property that a thing can have. It just means the two happen to be isomorphic. And it's not like it's some weird coincidence that they happen to be isomorphic, since a computer is always a physical object that utilizes physics to do its computing. This is why Deutsch instead claims that computation is a branch of physics. But many find this correct, in my opinion, answer unsatisfying, because it feels more like math than physics. I know Sadi [name spelling uncertain] has argued that one with me too. He says, look, everybody believes it's math, and it's treated like math, and you do proofs with it. It's math, Bruce, it's math, right?</p>

<p><em><strong>[00:33:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> You're talking about computation.</p>

<p><em><strong>[00:33:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes, computational theory.</p>

<p><em><strong>[00:33:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:33:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, it isn't surprising since it's literally, it is literally the study of what math we are able to do to physically compute.</p>

<p><em><strong>[00:33:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So yes, it's math.</p>

<p><em><strong>[00:33:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's like the intersection of math and physics, right?</p>

<p><em><strong>[00:33:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So it's both.</p>

<p><em><strong>[00:33:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Like there's nothing wrong with it being both. Nor is it physics in the normal straightforward sense that you're probably normally used to, where we don't use the LHC or other physics experiments of any kind to advance our knowledge in computational theory. We don't go out and build radio telescopes to try to advance our computational theory knowledge, right? It's treated way more like a mathematical discipline. Even the invention of the quantum computer required no physics experiments, right? I mean, Deutsch sat down and figured out how to map quantum phenomena to the Turing machine. And that was how he came up with quantum computational theory. It didn't require him to go get time with the LHC to work out computational theory. That just isn't what computational theory is, right? So that's why I think I would answer physics is more fundamental, even though I know that's a stupid answer, is because, and I think that this is why Deutsch, I think I'm going to do this in a future slide here that I'm getting to, but it's why Deutsch, Deutsch will immediately point out that there are objects that exceed the universal computer, like a universal constructor can do anything a universal computer can do, and then some, like it can construct things, right?</p>

<p><em><strong>[00:35:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's not really that surprising that there are objects in reality that have a greater repertoire than a computer, and the fact that they do doesn't undermine the Church-Turing-Deutsch thesis, and the fact that, like, if I were to, I've seen people argue this, they'll say, well, you know, if the universal constructor exceeds the repertoire of a computer, then that undermines the Church-Turing-Deutsch thesis. No, you just don't understand when you say that, right?</p>

<p><em><strong>[00:35:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Like, you've misunderstood.</p>

<p><em><strong>[00:35:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, but that's why I would, that's why I think I would answer that even just based on everything that we're talking about, physics feels more fundamental to me than computation. And so, and that's probably why I can never really get on board with Sardius' [name spelling uncertain] arguments, is because I'll always answer, no, neither is more fundamental. That's a dumb question. Let's not even ask that question. But like somewhere inside, there's a part of me that going, no, it's actually physics.</p>

<p><em><strong>[00:35:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Physics is more fundamental.</p>

<p><em><strong>[00:36:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Even though I know that's not true, right? But just as much as computation feels fundamental to her under CTD, CTD to me feels like physics is fundamental.</p>

<p><em><strong>[00:36:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> They're interwoven concepts.</p>

<p><em><strong>[00:36:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right, exactly.</p>

<p><em><strong>[00:36:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So, definition.</p>

<p><em><strong>[00:36:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is from page 12 of the book, Rucker's book. A computation is a process that obeys finitely describable rules. So, a computation is utterly deterministic or, in other words, non-random. The rules act like a kind of recipe for generating future states of that computation. Now, Rucker points out that describable is a slippery notion. Logicians have established that describable can't in fact have a formally precise meaning. Otherwise, a phrase like the following would be a valid description of a number.</p>

<p><em><strong>[00:36:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And now, here's the description.</p>

<p><em><strong>[00:36:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Let the Berry number be the smallest integer that can be described in less than 18 words. If such a number existed, then the definition of a Berry number, which was the definition was 17 words long, would in fact describe that integer in less than 18 words. So of course, this is just a form of Godel's paradox. So what we've really done is we've shown that the concept of describable is non-computable or it's equivalent to the halting problem, in other words. Now, many Deutschians make a strange argument that I need to dispense with before I continue. They will claim that AGI will not be an algorithm because an algorithm contains inputs and outputs and an AGI will not have an ending and won't have inputs and outputs. And I've heard this so many times and it's usually attributed to Deutsch. And I think at some point, I actually started to believe Deutsch had said it. But I actually looked it up for this podcast. And Deutsch not only has never said that, but he says the opposite of it. So I'll get the actual quote from Deutsch here in a second. But you'll hear this one all the time. And I don't know where it originated from.</p>

<p><em><strong>[00:38:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It is false.</p>

<p><em><strong>[00:38:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:38:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So the idea that the AGI program isn't an algorithm is just not true.</p>

<p><em><strong>[00:38:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> People think that Deutsch... Sorry, I might have spaced out for a second. They think that they think that Deutsch said that AGI is not an algorithm?</p>

<p><em><strong>[00:38:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes.</p>

<p><em><strong>[00:38:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:38:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> A program but not an algorithm.</p>

<p><em><strong>[00:38:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:38:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Oh, yeah, I had not heard that. I thought he was very clear on that.</p>

<p><em><strong>[00:38:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I've heard it from so many different people.</p>

<p><em><strong>[00:38:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:38:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So Rucker uses a humorous example of a calculator. Is a calculator, strictly speaking, an algorithm? It's not like the calculator freezes up at the end of its computation and you can never use it again. The computation that the pocket calculator is doing, it waits around to be used again. So I guess you could say it's not an algorithm, right? So Rucker says of this, this is page 17, this freezing up definition of halting is appropriate for certain simple models of computation such as an abstract device known as Turing Machines. But for more general kinds of computation, freezing up is too narrow a notion of a computation being done. So he gives examples. Go out and use Google Maps, okay, to find directions to your location. You may initially think of this computation as halting because it gives you a final result. But in reality, your web browser continually pulls for the mouse and for it continues to see if you're going to click on something. So the computation never halts, right? Your PC continually runs background processes, in fact. This whole, so this whole halting thing, which is part of the definition of an algorithm, a formal definition of an algorithm, really it's just a matter of convenience for humans to allow us to carve out parts of a program and think of them as having beginnings and ends for us to think of them in a certain way. So AGI will be a collection of algorithms just like any other program.</p>

<p><em><strong>[00:40:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:40:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, as I said, I don't know where this idea comes from, that AGI won't be an algorithm. And it's been attributed to Deutsch, but apparently wrongly so. So for example, in Deutsch's famous Aion article about AGI, on why we haven't discovered AGI yet, he says, so in this case, and actually in all other cases of programming, genuine AGI is only an algorithm with the right functionality would suffice. It's only an algorithm with the right functionality would suffice. So he refers to AGI as an algorithm there. He also elsewhere in the article refers to the software running on the human brain as the human's algorithm. So this idea does not come from Deutsch, okay? So hopefully we've done away with that, but keep this in mind. And I know that some of my audience may have this idea AGI is not an algorithm. We're going to now show that now that I'm trying to make sure you realize that's a false idea. Stop thinking of it that way. I think the only sense in which that's a true statement is that it's actually more like it's a collection of algorithms that run together, right? But programs that don't halt, like go play Skyrim.</p>

<p><em><strong>[00:41:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> There's no end to Skyrim, right?</p>

<p><em><strong>[00:41:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's not an algorithm in that sense, but it is a collection of algorithms. There's an algorithm where you take some sort of input, it updates its model of the world, and then it paints the screen to map what you see, okay? And you can think of that whole process as a single algorithm. It's important to realize that you can make that exact same trick with what humans do, okay? And I'm going to give examples of this as we go along.</p>

<p><em><strong>[00:41:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Rucker is going to give examples of this as we go along.</p>

<p><em><strong>[00:41:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So now there are functions that are computable, and there's functions that are not computable, the halting problem being the quintessential example of a non-computable function. But even among ones that are computable, many of them, even though you can precisely define what you want computed and how to compute it, you still may not be able to feasibly compute it.</p>

<p><em><strong>[00:42:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is of course the concept of intractability.</p>

<p><em><strong>[00:42:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:42:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Even though computations are deterministic, they can yield surprising results.</p>

<p><em><strong>[00:42:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is from page 20.</p>

<p><em><strong>[00:42:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That is to say, many computations are unpredictable because they yield surprising results. You couldn't have foreseen that outcome just by looking at the computation itself, looking at the algorithm itself.</p>

<p><em><strong>[00:42:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:42:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So let's try to define predictability a bit better.</p>

<p><em><strong>[00:42:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So P is predictable.</p>

<p><em><strong>[00:42:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is all Rucker still.</p>

<p><em><strong>[00:42:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If there is a shortcut computation Q that computes the same results as P, I say it's all Rucker, but he's quoting Wolfram, but very much faster.</p>

<p><em><strong>[00:43:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:43:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Let me read that again.</p>

<p><em><strong>[00:43:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> P is predictable if there is a shortcut computation Q that computes the same result as P, but much faster.</p>

<p><em><strong>[00:43:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Otherwise P is said to be unpredictable.</p>

<p><em><strong>[00:43:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This definition might make you uncomfortable at first, but it's actually self-evidently true if you stop and think about it. Let's use an example of a predictable computation. Let's say the orbits of the planets in the solar system. Let's say you want to predict the position of what the planets will be in a million years. Now, one way you could do that would be to create a simulation of the planets and then run it for a million orbits of the Earth around the sun, i.e.</p>

<p><em><strong>[00:43:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> a million years.</p>

<p><em><strong>[00:43:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But you'd not need to do that. You could do that and that would simulate it and then you could then know what the result would be and you could get your prediction that way. But you'd not need to do that because the orbits are periodic. There is in fact a shortcut computation that can get the result without having to do a full simulation point by point. Now, let's make the following assumption. So, assumption A is computation A has no shortcut computation that computes the same results. Okay, so this is an assumption about some given computation. Now, my question for you is this. Can you predict this computation without running the computation itself? If you can, you just violated the very assumption that this whole thing was based on. So the answer, logically speaking, must be, no, you can't predict this computation without actually running the computation. Any such computation that has no shortcut computation, by definition, must be unpredictable and surprising in its results.</p>

<p><em><strong>[00:44:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And so did you just answer the halting problem?</p>

<p><em><strong>[00:44:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> No.</p>

<p><em><strong>[00:44:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> No.</p>

<p><em><strong>[00:44:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> No.</p>

<p><em><strong>[00:44:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Nothing that groundbreaking.</p>

<p><em><strong>[00:44:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> The halting problem.</p>

<p><em><strong>[00:44:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> This has nothing to do with the halting problem. Okay, sorry.</p>

<p><em><strong>[00:45:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Actually, that's not true.</p>

<p><em><strong>[00:45:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It does have something to do with the halting problem. And in fact, I'm going to get to what it has to do with the halting problem in just a second.</p>

<p><em><strong>[00:45:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:45:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But it is not like I've solved the halting problem, right? I'm going to rely on the halting problem, if that makes any sense. Must a computation be complex to be unpredictable? The answer is maybe surprisingly no. There are many simple computations that have no shortcut, and thus are unpredictable and thus surprising. So from page 22 to 23, Rucker says, The notion of computer programs being unpredictable is surprising because we tend to suppose that being deterministic means being boring. Note also that since we don't feel ourselves to be boring, we imagine that we must be non-deterministic, and thus not at all like a rules-based computational system. Rucker now goes over Wolfram's classes of computation. This is a really interesting theory. Let me just say that this theory does have some problems. Rucker actually points out some of the problems. But we're going to go over the theory in detail because I feel like even though it's probably ultimately not quite right, it's like onto something, if that makes sense, verisimilitude. So class 1 would be a computation that enters a constant state. So no surprise at all. Class 2 would be that it generates repetitive or nested pattern like the orbits of the planets around the sun.</p>

<p><em><strong>[00:46:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So again, no surprise.</p>

<p><em><strong>[00:46:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Class 3 would produce a messy random looking crud. So no structure at all. But the results can't be predicted and they are surprising. Or class 4, they produce a gnarly, interesting non-repeating pattern. So even though there's obviously some sort of pattern, the pattern is surprising. So there's the structure and there's an obvious pattern just looking at it.</p>

<p><em><strong>[00:47:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> But the pattern is surprising.</p>

<p><em><strong>[00:47:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Note, you can't always be sure which class you are. So if you're class 4, like you can't be sure if you're class 4 or really class 2, let's say, but you just haven't repeated yet. And class 3 and 4 may be indistinguishable from each other at first too.</p>

<p><em><strong>[00:47:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:47:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So these aren't again, these aren't decidable classes.</p>

<p><em><strong>[00:47:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You can't just say, oh, this unless you've got a good explanation for it. We'll give examples where you can have a good explanation for it. You can't just look at the pattern produced by computation and say, oh, that's clearly a class three, that's clearly a class four. They're not really meant to be used that way. I think what we want to get it, take away from this is that these four classes exist, right? Not that we can formally tell which one's which, if that makes any sense. But I don't think anybody doubts that there are computations that fit into these four classes.</p>

<p><em><strong>[00:48:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:48:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It may also differ by input for a single computation. In fact, defining what with certainty... So, for example, I may have a single computation that with a certain input immediately produces a constant state, in which case, there's no surprise. But with a different set of inputs, it may turn into a gnarly, interesting, non-repeating pattern, which means it's class four instead.</p>

<p><em><strong>[00:48:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:48:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, in fact, defining with certainty which class of computation is, is in fact itself known to be incomputable, equivalent to the halting problem. Yet it isn't hard to see that these are useful if rough classifications. Now, up to this point, I pretty much entirely agree with Wolfram.</p>

<p><em><strong>[00:48:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is me talking now.</p>

<p><em><strong>[00:48:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But this next doesn't seem quite right to me, so we're going to dig into it a little bit further. Wolfram has something called the principle of computational equivalence, or the PCE, principle of computational equivalence. Almost all process, and it's defined as, almost all processes that are not obviously simple can be viewed as computations of equivalent sophistication.</p>

<p><em><strong>[00:49:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay, so I'm going to say that again.</p>

<p><em><strong>[00:49:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The PCE, the principle of computational equivalence, is that almost all processes that are not obviously simple can be viewed as computations of equivalent sophistication. Peter, run your bullcrap counter across that statement.</p>

<p><em><strong>[00:49:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Tell me what you think.</p>

<p><em><strong>[00:49:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Sorry, can you, you said it twice. Can you just say it one more time?</p>

<p><em><strong>[00:49:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Almost all processes that are not obviously simple, can be viewed as computations of equivalent sophistication. So if a computation isn't obviously simple, then all computations are equivalently, equivalently, equivalently sophisticated.</p>

<p><em><strong>[00:49:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Other than the.</p>

<p><em><strong>[00:50:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Am I crazy or is that a lot like Deutsch's principle of optimism?</p>

<p><em><strong>[00:50:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> You don't explain.</p>

<p><em><strong>[00:50:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Are all all problems, if they are interesting, are?</p>

<p><em><strong>[00:50:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Oh, no, I didn't see the connection at first, but you're right.</p>

<p><em><strong>[00:50:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> There is a connection there.</p>

<p><em><strong>[00:50:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Yeah, that's how I'm kind of reading it, which, you know, something I've given.</p>

<p><em><strong>[00:50:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> All right.</p>

<p><em><strong>[00:50:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, a lot of thought too.</p>

<p><em><strong>[00:50:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And apparently, your your bullcrap counter doesn't go off on the PC. Mine does.</p>

<p><em><strong>[00:50:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Like that.</p>

<p><em><strong>[00:50:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It seems to me like it's obviously false.</p>

<p><em><strong>[00:50:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I'm going to dig into it.</p>

<p><em><strong>[00:50:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's not as obviously false as my bullcrap meter thinks.</p>

<p><em><strong>[00:50:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> OK, OK.</p>

<p><em><strong>[00:50:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But but like to me, that just seems kind of ridiculous. This idea that think about in nature any any computation out there that's of any sophistication, that they're all equivalent.</p>

<p><em><strong>[00:50:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right.</p>

<p><em><strong>[00:50:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> How can that be?</p>

<p><em><strong>[00:50:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It just seems wrong to me.</p>

<p><em><strong>[00:50:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right.</p>

<p><em><strong>[00:50:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Just intuitively, it seems wrong.</p>

<p><em><strong>[00:50:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Fair enough.</p>

<p><em><strong>[00:50:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So this means that all complex computations, ones that you can't see immediately are simple, are equivalently complex. So Rucker gives the example of the motions of leaves on a tree to be a sophisticated computation, as sophisticated as the brain.</p>

<p><em><strong>[00:51:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:51:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So you got a tree, its leaves are moving in the wind, and that computation is as sophisticated as a human brain, doing, you know, running the mind.</p>

<p><em><strong>[00:51:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:51:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Doesn't that seem wrong to you?</p>

<p><em><strong>[00:51:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Like that seems wrong to me.</p>

<p><em><strong>[00:51:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right?</p>

<p><em><strong>[00:51:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, I think what Deutsch would, one thing I kind of get from Deutsch is that, that brains aren't, aren't all that complicated in some ways. We just, I mean, we don't understand them, because we don't understand that the program that is, but that is running on our brains, but we, you know, it's not like anything that crazy is happening, that couldn't be, be replicated on a, on a fairly simple computer.</p>

<p><em><strong>[00:52:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I can see I'm thoroughly failing to, to convince Peter that, that this has something intuitively wrong with it.</p>

<p><em><strong>[00:52:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Peter's got the exact opposite intuition as me.</p>

<p><em><strong>[00:52:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, I, you know, it could be, I'm just looking at it through my, my, use a controversial term, Deutschian lens, yeah, but yeah.</p>

<p><em><strong>[00:52:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You know, I, to some degree, I want to actually emphasize this. I do think human beings want to reason through their bullcrap meter, right? They want to use their intuitions. And kind of just subjectively say, no, that explanation seems silly to me. Oh no, that seems like it really explains things well. And one of the things I've really kind of emphasized in our last few podcasts is that that's the opposite of Popper's epistemology, right? Popper's epistemology is in some sense really about how do you squeeze out those subjective criticisms from the process and get down to things that we can all agree upon, what I've called objective criticisms. And the reason why is because our intuitions, as strong as we might fill them, they're just sometimes just really completely wrong. You can have a very strong intuition that you can't possibly be wrong about something and you can be wrong anyhow. And so I'm going to argue that the PCE is maybe closer to the truth than my intuitions suggest, okay?</p>

<p><em><strong>[00:53:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So let me say this.</p>

<p><em><strong>[00:53:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The reason why I feel like I don't buy the PCE is for the obvious reason that we can give a realistic simulation of a tree, of leaves on a tree, with far more limited computation than we could of a brain. To put this more plainly, I could today go into unity and make a tree blowing its leaves in the wind, and it would take X amount of computation, a very small amount, right? Whereas if I was trying to simulate Peter's brain in unity, I probably wouldn't have anywhere near the resources necessary to do it.</p>

<p><em><strong>[00:54:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right?</p>

<p><em><strong>[00:54:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Fair enough.</p>

<p><em><strong>[00:54:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So based on that simple example, it's certainly, my first impulse is something's wrong with the PCE.</p>

<p><em><strong>[00:54:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> The brain is more like a forest, maybe.</p>

<p><em><strong>[00:54:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Well, okay, so that would be an example.</p>

<p><em><strong>[00:54:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If a tree blowing in the wind is a sophisticated computation, why wouldn't a whole forest of trees be a more sophisticated, more complex computation? Again, that seems like it's really obvious, and it seems like it's a contradiction to the PCE.</p>

<p><em><strong>[00:54:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:54:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So let me take this a little bit further though, because maybe what's going on here is that I'm reading it too literally and I need to read it a little bit more charitably. So I think a charitable reading here would probably be something more like this. Okay, wise guy, I can imagine Stephen Wolfram saying this to me. Okay, wise guy, but can you give an exact simulation of the leaves blowing in the wind on a tree without actually having a specific tree with a specific wind on a specific earth with a specific environment? And I think that's really probably what Wolfram's trying to get at here. Okay, because I think the answer to that is no, I can't. Okay, so I would concede the point at least that much. This is where Wolfram's theories really turn out to not simply be a version of the Church Turing-Deutsch thesis. The reason why is because the Church Turing-Deutsch thesis says we can simulate leaves blowing on a tree realistically, not we can simulate an exact tree in an exact situation. In fact, I tend to agree with Wolfram here that it is presumably simply impossible to simulate an exact tree in an exact environment with an exact wind without basically having that exact tree in the exact environment with an exact wind. If I do accept that view, and I'm still not sure if I do or not, there are some consequences. One of the consequences is called, this is from page 27 now, the principle of computational unpredictability, PCU, which is most naturally occurring complex computations are unpredictable, where complex means either class 3 or class 4.</p>

<p><em><strong>[00:56:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> i.e.</p>

<p><em><strong>[00:56:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> there simply is no shortcut for that specific computation, and so it gives surprising results that can't be predicted except by running the full computation itself. And to quote Rucker directly, it follows that for many systems, no systemic prediction can be done, so that there is no general way to shortcut their process of evolution, and as a result, their behavior must be considered computationally unpredictable. Though I feel the PCE is not quite right in some way, I feel like it's making a point that does lead to a correct idea in the PCU, namely, that the only way to predict most computations is to actually do them. However, Wolfram goes too far, in my opinion, or at least depending on how you read him, when he goes on to say, and this, I believe, is the fundamental reason that traditional theoretical sciences has never managed to get far in studying most types of systems whose behavior is not ultimately quite simple.</p>

<p><em><strong>[00:57:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That is a quote from Wolfram.</p>

<p><em><strong>[00:57:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I'm quoting him from Rucker's book on page 28.</p>

<p><em><strong>[00:57:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Wolfram needs to take a harder look at the Church–Turing–Deutsch thesis here. The distinction between simulating a kind of system follows from the CTD, as opposed to simulating a specific system which does not follow from the CTD. And that difference is deeply relevant here. We will never simulate most systems.</p>

<p><em><strong>[00:58:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's Wolfram's point.</p>

<p><em><strong>[00:58:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But we can understand the simple computations that create those complex, unpredictable and surprising outcomes.</p>

<p><em><strong>[00:58:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's the Church–Turing–Deutsch thesis' point, in my opinion.</p>

<p><em><strong>[00:58:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[00:58:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And maybe we could see them as not at odds with each other if we interpret them in this way.</p>

<p><em><strong>[00:58:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I do think that that's probably a slight tweak from what Wolfram originally said or meant. Maybe even a slight enough one, but one that has profound consequences. But that's my opinion on it, that if we can accept this tweak, then I can maybe accept Wolfram's point of view here. So to some degree, Wolfram does seem to get this. From page 106, we have the bad news that Wolfram brings for physics is that in any physically realistic situation, our exact formulas fail and we're forced to use step-by-step simulations.</p>

<p><em><strong>[00:59:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay, that's true.</p>

<p><em><strong>[00:59:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We can't just run the computation and get the result. We have to actually simulate every atom moving in relationship to every other atom. So a real object's motion, this is page 106, a real object's motion will at times be carrying out a class IV computation. So in a formal sense, the object's motion will be unpredictable, meaning that no simple formula can give full accuracy. Now, from page 28, when a computation generates an interesting and unexpected pattern of behavior, we call this emergence, which I think is a really great way to understand emergence. Emergence is when a computation generates an interesting and unexpected pattern of behavior. An example would be the Mandelbrot set. It's a famous example of emergence from simple rules that have infinite complexity. But we can immediately see it isn't just random. It forms rough non-periodic patterns, making it a class IV pattern, i.e. it produces gnarly, interesting, non-repeating patterns. Another famous example of emergence is the interesting non-repeating patterns of cellular automata. And this is really what Wolfram has kind of made himself famous for, was his studies into cellular automata. In fact, he has made wild claims about physics really being cellular automata and using cellular automata to recreate our physics theories. These are challenged, and I don't think I agree with him on that front. But the concept of cellular automata are interesting.</p>

<p><em><strong>[01:00:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Go look that up.</p>

<p><em><strong>[01:00:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'm not going to get into what cellular automata are, but go look it up. It's usually a very simple set of rules where you turn on pixels or turn off pixels. And the rules give surprising results that create pretty pictures and interesting looking patterns and things like that that you could never have predicted from the simple rules.</p>

<p><em><strong>[01:01:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, I was just looking up the... I'm one step behind here looking at the Mandelbrot set on ChatGPT here. And so this is about fractals, right?</p>

<p><em><strong>[01:01:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes.</p>

<p><em><strong>[01:01:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Actually, Google the Mandelbrot set and just look at a picture of it.</p>

<p><em><strong>[01:01:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Oh, okay.</p>

<p><em><strong>[01:01:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's beautiful.</p>

<p><em><strong>[01:01:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:01:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's very interesting looking.</p>

<p><em><strong>[01:01:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And this guy, this is someone who invented the concept of fractals? Or is just this one example of what a fractal is?</p>

<p><em><strong>[01:01:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah, no.</p>

<p><em><strong>[01:01:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So the Mandelbrot, I can't remember the story of the Mandelbrot set. Like, it seems like it was they had were running a little program and they decided to draw a picture of it on the screen and immediately saw that it was this really interesting looking picture.</p>

<p><em><strong>[01:01:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And it was surprising.</p>

<p><em><strong>[01:01:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Like, have you looked it up?</p>

<p><em><strong>[01:01:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Like, look it up and take a look at it.</p>

<p><em><strong>[01:01:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:01:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, here's the thing that's cool about it. If you zoom in on any of the details in the Mandelbrot set, the zoomed in version is just as repeatedly interesting.</p>

<p><em><strong>[01:02:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:02:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It all kind of looks roughly similar, but you can tell it's not the same, right?</p>

<p><em><strong>[01:02:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I see.</p>

<p><em><strong>[01:02:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You zoom in on the details and it turns out that what looked like simple detail, once you get close enough, it starts to turn into this interesting set of details itself that kind of look like the Mandelbrot set, and you can immediately tell I'm still looking at the Mandelbrot set because it's so similar, and yet it's not the exact same thing, right?</p>

<p><em><strong>[01:02:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:02:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Because it's infinitely surprisingly interesting. There's an obvious pattern there, but the pattern is not periodic.</p>

<p><em><strong>[01:02:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:02:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> So this is a kind of fractal, that not all fractals are Mandelbrot?</p>

<p><em><strong>[01:02:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's right.</p>

<p><em><strong>[01:02:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's right.</p>

<p><em><strong>[01:02:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:02:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Interesting.</p>

<p><em><strong>[01:02:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So this is why Wolfram tries to work out how to drive physics from something like cellular automata. He's trying to fit physics into a framework of a class 4 computation. Now, I personally have my doubts that this is the right way to go about it. And honestly, it doesn't seem like that even is an implication of his theory. Compare this to how I accept the Universal Explanership hypothesis, but feel [word uncertain] Deutsch has derived several theories from it that aren't actually implied by the theory. I feel the same way here, that I agree with the basics of what Wolfram is saying, but it does not at all seem to me that we should try to force fit physics into the framework of a cellular automata.</p>

<p><em><strong>[01:03:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Even if you accept his theory as correct, I just don't think that's an implication that makes sense to me.</p>

<p><em><strong>[01:03:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now from page 30, emergence is different from unpredictability. On the one hand, we can have unpredictable computations that don't have any high-level emergent pattern. The dull digits of pi would be an example of this. On the other hand, we can have computations that generate emergent patterns that are in the long run predictable. So this is all from page 30, that was all Rucker. So Rucker gives these examples, and I can't pronounce all these. The Vichniac Vote Rule, ultimately predictable and thus class 2. The flocking behavior, seeing flocking behavior with like birds. Usually class 4, we think, but sometimes class 2. The Mandelbrot, assumed to be class 4, but no way to prove it. You kind of look at it, you can kind of immediately tell it's class 4. It's almost the quintessential example of a class 4, but there's no way to prove it.</p>

<p><em><strong>[01:04:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:04:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So from page 43, a computation is universal if it can emulate any other computation.</p>

<p><em><strong>[01:04:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is an interesting point.</p>

<p><em><strong>[01:04:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, I had a CritRat friend that thought that the fact that the human brain was equivalent to a universal computation was a stunning revelation. He wanted to go around teaching children and people, you have a universal computer in your brain.</p>

<p><em><strong>[01:04:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Isn't that amazing?</p>

<p><em><strong>[01:04:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right?</p>

<p><em><strong>[01:04:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The problem is that nearly every computation is universal. In fact, it's so difficult to come up with non-universal computational machines that you have to carefully stop and think about how to make something like a finite automata or a push down automata such that they're not equivalent to a Turing machine. The vast majority of computing machines that you will invent, if you just like go out and just invent one, it will be a Turing machine, like it will be equivalent to a Turing machine. It won't be equivalent to a finite automata because the vast majority of imaginable computations are universal.</p>

<p><em><strong>[01:05:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Yeah, when I first read Beginning of Infinity, I thought that that was pretty much David Deutsch's point, was that human brains are universal computers. But now I realize that universal computers are a dime a dozen.</p>

<p><em><strong>[01:05:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> They are.</p>

<p><em><strong>[01:05:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And that what he's really saying is that it's something, it's a universal explainer.</p>

<p><em><strong>[01:05:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Which is a different thing.</p>

<p><em><strong>[01:05:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> That has some, and it's more that he's using the universal computer thing as more of an analogy, I guess. That's how I'm currently understanding it.</p>

<p><em><strong>[01:05:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> As we've discussed on this podcast, there are some connections between the two concepts, which I think further confuses people, but they are not the same concept.</p>

<p><em><strong>[01:06:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And it is more of an analogy.</p>

<p><em><strong>[01:06:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:06:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:06:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> From page 43, quoting Rucker, when we examine the naturally occurring computational systems around us, like air currents or growing plants or even drying paint, there seems to be reason to believe that the vast majority of these systems support universal computation.</p>

<p><em><strong>[01:06:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That may be surprising to many.</p>

<p><em><strong>[01:06:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:06:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you're ever curious, go look up the pool table computer. You can use a pool table to build a universal computer, right? Like universal computers are literally a dime a dozen. As such, we have every reason to believe that animal brains, especially ones with a neocortex like mammals, have universal computers for brains as well as humans. Now, of course, don't confuse this with universal explainership, which is something different, as you just said.</p>

<p><em><strong>[01:07:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:07:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, we might take the PCE, which is almost all processes that are not obviously simple, can be viewed as computations of equivalent sophistication, and we might take it to mean something like this. Most naturally occurring complex computations can emulate each other.</p>

<p><em><strong>[01:07:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is from page 49.</p>

<p><em><strong>[01:07:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> In fact, given this, that nearly all computations are universal, this must be correct. And with my small tweak that I've suggested, that we're talking about a specific emulation of something, not a general emulation of a class of something, we also have every reason to believe that nearly everything in nature has a huge amount of computational power to it. This gives rise to a reformulation of the PCE as most naturally occurring complex computations are universal. And thus, a reformulated PCU as most naturally occurring complex computations are unpredictable from page 43. Now, on page 87, Rucker actually references the three-body problem, made famous by the show The Three-Body Problem or The Books, as an example of how it is impossible even for a mere three bodies to predict the outcome of a computation without simply actually running the computation in real life. This was the basis for the now famous Three-Body Problem storyline on Netflix, where there are aliens that can't make good plans for their civilization due to being near three suns that make their planet's orbit wholly impossible to predict. Or maybe it was two suns because the planet was the third body.</p>

<p><em><strong>[01:08:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I can't remember.</p>

<p><em><strong>[01:08:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It doesn't matter.</p>

<p><em><strong>[01:08:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> So basically, if there's two planets, it's completely predictable. But if there's three, you've just got to run the simulation.</p>

<p><em><strong>[01:08:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> There's no way to compute it [ending wording uncertain].</p>

<p><em><strong>[01:08:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Is that fair? [wording uncertain]</p>

<p><em><strong>[01:08:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Obviously, we have multiple planets in our solar system, and they're completely predictable. But the reason why is because the sun is absolutely trounces everything else in terms of its gravity.</p>

<p><em><strong>[01:09:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Oh, okay.</p>

<p><em><strong>[01:09:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So with the three-body problem, the aliens lived, I think it was with three suns.</p>

<p><em><strong>[01:09:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And this isn't just about planets. This is just a statement about physical reality, right?</p>

<p><em><strong>[01:09:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right.</p>

<p><em><strong>[01:09:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So they were equally sized suns, and so their planet was bumping between them. And they didn't have a stable orbit. They would orbit around one sun, then they would get sucked away by another sun. And so it was basically impossible to predict what was going to happen because the three-body problem is thoroughly unpredictable without actually doing it in nature. That's the whole basis for the storyline of these aliens and the three-body problem and why they want to invade Earth.</p>

<p><em><strong>[01:09:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Did they even explain that on the show?</p>

<p><em><strong>[01:09:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I had like a cool graphic where they showed it.</p>

<p><em><strong>[01:09:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[01:09:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Because I need to watch it again.</p>

<p><em><strong>[01:09:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[01:09:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I thought they did a pretty good job of explaining it.</p>

<p><em><strong>[01:09:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So okay.</p>

<p><em><strong>[01:09:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:09:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Page 104.</p>

<p><em><strong>[01:09:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Even our most highly parallel digital computers have a minuscule number of computational nodes compared to nature.</p>

<p><em><strong>[01:10:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That was, I think, a quote from Rucker.</p>

<p><em><strong>[01:10:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So this kills the hypothesis that we're living in a simulation based on the supposed logic that most living beings will live in simulations. It's actually impossible to build a computer, at least according to the laws of physics as we currently understand them, absent something like, say, an Omega Point computer. It's impossible to build a computer that can simulate the whole universe. It would take a computer much larger than the whole universe. So here I am ignoring a special sort of computation like the Omega Point at the moment. But at the moment, the Omega Point is a discredited theory, so I feel okay to ignore it.</p>

<p><em><strong>[01:10:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:10:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So Rucker says, Digital computers have no hope of feasibly emulating the full richness of the physical world in real time.</p>

<p><em><strong>[01:10:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:10:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, you know, there was an interesting discussion I had recently. I quoted Pedro Domingos. He's a famous professor, computation, AI, that sort of thing. And he made a quote. I should have looked it up for the show, but he made a quote that I quoted, where he basically made the argument I just made, making fun of people who believe that we live in a simulation. Because it's basically based on our understanding of physics. It's impossible to make a simulation that simulates the whole universe. And therefore, this whole idea that most people will live in a simulation, it just isn't true because each simulation has much, much, much reduced resources compared to the outer world that the simulation is running in. Okay, so it just isn't the case. Like the whole argument that Bostrom makes is based on this completely wacky, unfounded, ridiculous idea. Okay, every crit rat that responded to me immediately said, no, that's a terrible argument that he's making because I can just imagine that we're in a simulation. And that simulation, but the outer world that we're running in has 10 to the 10 to the 10 to the 10 to the 10, you know, number of computations as the simulation. Here's the problem with that response and why that response is itself correct, but completely misses the point.</p>

<p><em><strong>[01:12:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What Pedro is doing is he's forcing you to think about your assumptions. He's pointing out that to make your simulation hypothesis work, you can't simply look at the world you're in and then say, look, in this world, we should assume that most people will live in simulations based on our current understanding of the laws of physics, and therefore we should assume that we're in a simulation.</p>

<p><em><strong>[01:12:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay?</p>

<p><em><strong>[01:12:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That does not follow from the laws of physics as we understand them. In fact, the opposite follows from the laws of physics as we understand them.</p>

<p><em><strong>[01:13:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay?</p>

<p><em><strong>[01:13:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> To be able to make that, you're actually throwing in an extra assumption, which is the one the crit rats were all trying to quote to me. You're actually saying there exists a world that isn't our world and doesn't follow our laws of physics and is different from our world. It has 10 to the 10 to the 10 to the 10th additional level of computation compared to the real world universe. This other world, if you first posit its existence based on no reason at all, not based on trying to solve a problem, not based because you needed it as part of an explanation, it makes no predictions.</p>

<p><em><strong>[01:13:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's a purely supernatural belief.</p>

<p><em><strong>[01:13:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:13:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you're willing to make that additional assumption, sure, then Nick Bostrom's argument now makes sense, but only under that circumstance.</p>

<p><em><strong>[01:13:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:13:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Once we realize that's the case, that's how you actually dismiss his argument. Is, okay, your argument is in fact starting with a totally supernatural assumption. And if you don't start with that assumption, if you start with the assumption that the laws of physics actually apply, then what you're saying doesn't make sense and your whole argument falls apart.</p>

<p><em><strong>[01:14:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And that was what I liked about his argument.</p>

<p><em><strong>[01:14:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Um, okay.</p>

<p><em><strong>[01:14:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, page 110, Rucker says, Wolfram also speaks of such unpredictable computations as irreducible or as intrinsically random. Now, Rucker takes issue, and so do I, with the wording, with wording that say, um, that wording saying that was [restart wording uncertain] because more technically it should be pseudorandom. Because what he's calling random here, intrinsically random here, comes out of a fully deterministic but unpredictable process. Note, and I'm going to get to this in a second, I think Rucker has also misused the term pseudorandom here. Because an unpredictable process is not necessarily pseudorandom. Now, as far back as episode seven, we talked about how there are two kinds of probability. Probability due to an actual random process, say predicting the rolls of a six-sided die, and probability due to simply being ignorant, say predicting a hidden six-sided die that has already been thrown. So it now definitely has a side that's up, and you just don't happen to know what it is.</p>

<p><em><strong>[01:15:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:15:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We have something similar going on here. We utilize the same probability calculus for both kinds of situations. Probability due to actual randomness, probability due to ignorance. And the probability calculus can be used in both circumstances.</p>

<p><em><strong>[01:15:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I will explain how I know that.</p>

<p><em><strong>[01:15:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That is absolutely the case.</p>

<p><em><strong>[01:15:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It can also be abused.</p>

<p><em><strong>[01:15:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I know Deutsch gets very vehement on this. And in fact, I'm going to argue so strongly that it almost turns into a hatred of the word probability. And I probably need to explore that further in a separate podcast. Okay, yes, using probability calculus for ignorance can turn out to be an abuse of the probability calculus, but it isn't always an abuse of the probability calculus. Okay, let me just say that for now.</p>

<p><em><strong>[01:16:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:16:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> That really distills things.</p>

<p><em><strong>[01:16:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[01:16:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So one argument is sometimes here, is that really the two kinds of randomness are the same. Both are really lack of knowledge of initial conditions via chaos theory. Now, quantum mechanics has strongly challenged this view. I would strongly challenge, it has refuted this view and suggested, at least from the point of view of a conscious being living inside of a universe, which is all of us, that randomness is fundamental to reality. So yes, I know Deutsch has several talks where he seems to claim otherwise. Let me actually, in just a moment, quote him and I will explain why what I'm saying is not at odds with what he is saying, and why I actually take issue with the way he words things, because I feel it causes misunderstandings. I would also note that most crit rats I've talked to about this seem to fundamentally misunderstand the notion of randomness due to these talks from Deutsch. For example, a deterministic multiverse where one sixth of all universes each get a different die roll is not at odds with the concept of randomness, but it's actually an explanation of how randomness is fundamental to reality from the point of view of an observer. Obviously, as an observer, some version of you ends up in each universe, but that is the same as saying that from your point of view, there is a one in six chance the die will come up on any one of the sides. Your meaning, whoever it is that is the one that sees that side. And presumably, we invented the term randomness and probability to explain such observations from the point of view of an observer in a single universe.</p>

<p><em><strong>[01:18:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Therefore, QM is not at odds with the concept of randomness or probability. In fact, it ties it deeply into reality in such a way that it is fundamental to reality. That randomness is a real thing and probability is a real thing, okay? Let me come back to that point, because I understand why people get confused on this. And I understand what Deutsch is trying to say, but he says it in a way that I feel is a little misleading. So let me actually hold the quotes to help explain here. So here is a quote from Deutsch from Fabric of Reality. He says, it is perhaps worth stressing the distinction between unpredictability and intractability. Unpredictability has nothing to do with the available computer resources. Classical systems are unpredictable, or would be if classical systems existed, because of their sensitivity to initial conditions. Quantum systems do not have that sensitivity, but are unpredictable because they behave differently in different universes, and so appear random in most universes. So notice how Wolfram refers to any classical unpredictability as randomness, and that this is misleading, because that's not what the term randomness normally means.</p>

<p><em><strong>[01:19:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So score one for Deutsch here.</p>

<p><em><strong>[01:19:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But also notice that Deutsch calls classical unpredictability intractability and reserves the term unpredictability only for true randomness. Now, this is also a misleading use of terms as an intractable algorithm is unpredictable. So you can't reserve that term just for randomness like Deutsch is trying to do. So score one for Wolfram here. So neither gentleman uses terms in such a way that we aren't likely to get confused. I find both of their use of terms very, very, very confusing.</p>

<p><em><strong>[01:20:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:20:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Let me make some suggestion.</p>

<p><em><strong>[01:20:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, then Rucker also uses terms in a way that I find confusing. Rucker tried to use the term pseudo random to refer to any unpredictable process.</p>

<p><em><strong>[01:20:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Here's the actual quote.</p>

<p><em><strong>[01:20:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I might mention in passing that computer scientists also use the word pseudo random to refer to unpredictable processes.</p>

<p><em><strong>[01:20:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:20:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That seems pretty clear.</p>

<p><em><strong>[01:20:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is not actually how the term pseudo random is normally used. It is not normally used as equivalent to unpredictable. Now, Rucker does go on and kind of explains this. So I don't know if he's really confused. I think it's just that one sentence is confusing the way it's worded. So he explains it like this. He says, any programming environment will have built in to it some predefined algorithm that produces reasonable random looking sequences of numbers. These algorithms are often called pseudo randomizers. The pseudo refers to the fact that these are in fact deterministic computations.</p>

<p><em><strong>[01:21:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is also from page 110.</p>

<p><em><strong>[01:21:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> An unpredictable algorithm may or may not produce a reasonable random looking sequence. Isn't that kind of Rucker's point, that it may produce some sort of pattern that is clearly not random?</p>

<p><em><strong>[01:21:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:21:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Isn't that the difference between a class 3 and a class 4 computation under Wolfram's classifications? It's a mistake to try to equate unpredictability with pseudo randomness. Specifically, pseudo randomness is a kind of unpredictability, but unpredictability is not always pseudo random, would be the correct or at least the less confusing way to word things. However, you can use a fully deterministic process and treat it identically to being a stochastic and random, truly a stochastic, that is to say truly random process. The fact that you can do that is notable.</p>

<p><em><strong>[01:22:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:22:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is part of what I'm trying to explain.</p>

<p><em><strong>[01:22:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is why Turing machines don't require a randomizer to be considered a universal computer. If stochastic processes or random processes were necessary for some algorithms to work, then the Turing machine wouldn't be a universal computer. You would have to be a Turing machine with a randomizer attached would be necessary to have a universal computer.</p>

<p><em><strong>[01:22:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That just isn't the case, right?</p>

<p><em><strong>[01:22:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Like literally every algorithm that you can imagine with a randomizer, you can do it with a pseudo randomizer and it will still work.</p>

<p><em><strong>[01:22:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:22:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Here's something.</p>

<p><em><strong>[01:22:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is going to be a controversial statement, but it just follows from everything we're talking about. This is why Bayesians are correct, or at least sometimes correct, is what I mean, to treat ignorance as probability. Because the probability calculus does apply to pseudo random numbers, even though it's a fully deterministic non-random process. This is why, when you try to claim probability calculus doesn't apply to ignorance, which is what I've seen many crits rats try to claim and often quoting Deutsch, maybe not understanding Deutsch, that they've missed something. If I go run a computer program and I have a pseudo random number generator, it's a deterministic process that's been intentionally made to look like a random pattern. It's surprising in exactly the same way a random pattern would be.</p>

<p><em><strong>[01:23:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Almost the same way.</p>

<p><em><strong>[01:23:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The way you would deal with that would be with the probability calculus and it would work correctly even though it is not actual randomness. This means that the Bayesians are correct, at least in some circumstances, to apply probability to non-random processes. And instead, think of it as a form of ignorance. If you don't believe that, you are the one that's wrong and the Bayesians are right. The correct way to go about criticizing the Bayesians isn't to try to deny that very essence. It's so easy to go show it's right by writing a pseudo-randomizer. Like it's easy to refute your view if you think they're wrong altogether. What you really should be criticizing is when they use it in cases where it isn't a pseudo-random process.</p>

<p><em><strong>[01:24:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is a topic for another time.</p>

<p><em><strong>[01:24:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:24:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So the Bayesians do get something wrong here.</p>

<p><em><strong>[01:24:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It just isn't what the crit rats think it is.</p>

<p><em><strong>[01:24:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Also note that Deutsch claims that there is no such thing as true randomness. Let me quote here again, just so you can be clear that he did say this. He says, from Fabric of Reality, quantum systems do not have that sensitivity, but are unpredictable because they behave differently in different universes and so appear random in most universes. Notice the wording, so appear random. This is also terribly misleading. That would only have existed had quantum physics not been many worlds. And that what we call randomness is actually only apparent randomness to him. But I take issue with this misleading way of wording things. It's not that I think he's conceptually wrong here, okay? But presumably the word random was invented by people living in single universes to explain why some things were fundamentally unpredictable from their point of view, from their point of view in a single universe. So insisting that this kind of unpredictability only appears random, forcibly assigns the word random a very strange meaning of a non-existent thing that only can appear in a quantum universe that doesn't have many worlds, that doesn't exist in real life, but happens to be identical in every way to how individual universes work from the point of view of a single observer. The reason why I raise this is I have had numerous crit rats tell me that there's no such thing as probability because of many worlds. It's such a basic misunderstanding of what the term probability normally means, right?</p>

<p><em><strong>[01:26:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Of [restart wording uncertain] quantum physics is fundamentally probabilistic from the point of view of an observer. The fact that it's not probabilistic at the level of the multiverse, that's also true, but observers don't exist at the level of the multiverse.</p>

<p><em><strong>[01:27:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So it doesn't matter.</p>

<p><em><strong>[01:27:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It doesn't matter that there's no randomness at the level of the multiverse. Quantum physics means that there is randomness at the level of an observer in a universe.</p>

<p><em><strong>[01:27:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Period.</p>

<p><em><strong>[01:27:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> End of story.</p>

<p><em><strong>[01:27:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This makes probability fundamental to reality according to quantum mechanics. So let me just back up and let me make my own suggested use of terms, which I think will be more clear across the board and will avoid a lot of this confusion. So let's define intractability as any process or computation that has no shortcut computation.</p>

<p><em><strong>[01:27:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I think that's a good way of understanding intractable.</p>

<p><em><strong>[01:27:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Let's define random as any process that is non-deterministic and thus stochastic, at least from the point of view of an observer in a universe. This is presumably what the term originally meant anyhow, and I just don't see the need to forcibly assign it a non-existent meaning. That would strike me as the essentialist mistake anyhow. Pseudorandom is a fully deterministic process that is unpredictable due to its intractability, but outputs a spread similar to what a true random process would look like. Note that not all unpredictable processes are pseudorandom, but all pseudorandom processes are unpredictable. And finally, unpredictable, we're going to define as any process that is either random or intractable, and thus cannot be predicted. And they may or may not be pseudorandom. Okay, with that out of the way, which I think is a way more clear way of wording things that just gets past a lot of the misunderstandings I keep hearing.</p>

<p><em><strong>[01:28:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Let's get back to the regular schedule program.</p>

<p><em><strong>[01:28:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So Rucker argues that PCE and PCU are just conjectures and not mathematical proofs.</p>

<p><em><strong>[01:28:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> He says that on page 387.</p>

<p><em><strong>[01:28:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> He states multiple times that he does not agree with the PCE, principle of computational equivalence.</p>

<p><em><strong>[01:28:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Instead, he suggests a weakened form of it that he calls the natural unsolvability hypothesis, or he calls it the NUH, which can be perhaps mathematically proven. Without going into a full proof, let me give you a high level sketch of what Rucker has in mind. So, recall that a human mind being a computation can be thought of as an algorithm. Okay, that's why I spent a lot of time making sure it was clear that Deutsch did not claim that the human mind wasn't an algorithm. Example, suppose Peter is building a model plane. This act of Peter building a model plane is an algorithm, because it has an identifiable input and a checkable final state. Now, you may not be used to thinking of something like this as an algorithm, but this is what we mean by algorithm, normally speaking. Now, of course, Peter himself is not an algorithm, in the sense that what we call Peter is just some computation that halts and then it's done. Though neither is nearly any computation you work with, including your pocket calculator. So, it's more a matter of how we talk about programs. It's convenient to slice them up into starting and ending states, and we conveniently call this an algorithm. One use of your pocket calculator is an algorithm, and Peter building a model plane is also an algorithm.</p>

<p><em><strong>[01:30:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> What does any of this have to do with predictability?</p>

<p><em><strong>[01:30:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Let's say I have a little device that could predict with 100 percent accuracy whatever Peter was going to do. So, if I ask this device, if Peter will someday be the CEO of Google, it could tell me with 100 percent accuracy if he would or would not be the CEO of Google. But, Peter is a universal computation. So, this device must be able to solve the halting problem.</p>

<p><em><strong>[01:30:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's impossible.</p>

<p><em><strong>[01:30:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Thus, this little prediction device must not exist. Rucker argues that this is really what Wolfram, whether he realizes or not, is trying to argue. That for a universal computation to be predictable, it would have to also violate computational theory.</p>

<p><em><strong>[01:31:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> But wait, this is not a very good mathematical proof.</p>

<p><em><strong>[01:31:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> For one thing, it assumes nearly all computations in nature are universal, which we of course know not all of them are. Now, this is one of Wolfram's assumptions, but why should you buy it? Rucker shows that even non-universal computations commonly have undecidable questions for them. Sure, if the computation is really simple, say, always terminates with an answer of 1, then you could solve the halting problem for that particular computation. You could say, oh, it always halts with the answer of 1. But the moment a computation starts to get complicated, it either automatically becomes universal, and thus the argument above applies, or it does not, but it starts to have its own lesser version of the halting problem to deal with.</p>

<p><em><strong>[01:32:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So this lines up nicely with Wolfram's assumptions that there are four classes of computations, doesn't it?</p>

<p><em><strong>[01:32:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> With class 1 and class 2 being predictable, and class 3 and class 4 being unpredictable, even if you can't prove which class of, you can't prove them universal, or even that they are not universal. So a corollary from Rucker's NUH is mostly naturally occurring complex computations are runtime unbounded relative to some target detector algorithm.</p>

<p><em><strong>[01:32:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is from page 423.</p>

<p><em><strong>[01:32:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Example, am I doing the right thing to write a best seller?</p>

<p><em><strong>[01:32:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Answer.</p>

<p><em><strong>[01:32:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But wait, this isn't a very good mathematical proof. For one thing, it assumes that nearly all computations in nature are universal. And as we just said, not all of them are. Now this is one of Wolfram's assumptions, but why should you assume that? Sure, if the computation is really simple, say let's say it always terminates with an answer of one, then you can solve the halting problem for that particular computation.</p>

<p><em><strong>[01:33:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It obviously always halts with the answer of one.</p>

<p><em><strong>[01:33:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But the moment a computation starts to get complicated, it either automatically becomes universal, and thus the above argument applies, that to predict the outcome would violate the halting problem, or it does not become universal, but it starts to have its own lesser version of the halting problem. So this is the main thing that Rucker is trying to explain, is that even a non-universal computation, unless it's something like it always terminates with one, has its own undecidability problem. And therefore, even if you want to start with the assumption that not every single computation is universal, because not all of them are, still the vast majority of them would have a decidability problem and thus have the halting problem on them. So this lines up nicely with Wolfram's assumption that there are four classes of computations, doesn't it? With class one and class two being predictable, and class three and four being unpredictable, even if you can't prove that they're universal. And even if they're not universal, this still lines up well with his classes. So now a corollary to Rucker's NUH is from page 423. Most naturally occurring complex computations are runtime unbounded relative to some target detector algorithm.</p>

<p><em><strong>[01:34:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> What does that mean?</p>

<p><em><strong>[01:34:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> An example of that is, if you were to ask the question, am I doing the right thing to write a bestseller? The answer, computationally speaking, is you must wait to see if your book is a bestseller. Because if you could answer the question definitively, then you would have a target predictor that would be equivalent to solving the halting problem, which we know is impossible.</p>

<p><em><strong>[01:35:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:35:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, all of this, let me see if I can kind of summarize everything that we've set up to this point using Wolfram's theories. Nature is, everything in nature is a terribly large, sophisticated, complicated computation. And everything in nature is connected to everything else. Those, that wind on the leaves, it's tempting to say it's a smaller computation than the rest of the forest. But the computation of the specific wind on that specific tree would require you to compute the rest of the forest to be able to figure out what specific air currents are coming to that particular tree. That's why the sophistication of the tree with its leaves on the wind is equivalent of sophistication to the entire forest with its leaves on the wind.</p>

<p><em><strong>[01:36:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Because you have to have basically almost everything to take into consideration a specific tree with a specific leaves.</p>

<p><em><strong>[01:36:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Once you realize that's the case, and once you realize that almost all computations are universal, and even the ones that aren't still have a decidability problem, you can start to understand why it is that entirely deterministic computations are by nature, almost always unpredictable, by definition, okay? And why really when people start getting confused, start getting upset over, oh no, the human mind is deterministic, that destroys free will because then we're completely predictable. Like the whole thing is just a misunderstanding, right? Like all every ounce of it is a misunderstanding of even the very concept of predictable. What they really mean is that once you run the computation, you know the outcome. And if you were to run the same computation, you would then know the same outcome, okay? But that never happens in nature. It's not even possible for it to happen in nature, okay?</p>

<p><em><strong>[01:37:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> So reality is algorithmic. But what I'm kind of interpret you saying is because everything is interrelated, that the something like the halting problem from that perspective, like reality is just like a big halting problem or something.</p>

<p><em><strong>[01:37:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yes.</p>

<p><em><strong>[01:37:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It seems to-</p>

<p><em><strong>[01:37:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Decidability, the idea that you could predict the outcome of reality would be equivalent to saying you've solved the halting problem. Therefore, we know it's impossible that you can predict the outcome.</p>

<p><em><strong>[01:37:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[01:37:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> That's a really compelling way to think of it.</p>

<p><em><strong>[01:37:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Wow.</p>

<p><em><strong>[01:37:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:37:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, let's back up a little bit now and let's talk about the two Bob's problem. Except that it's not a problem, of course. What you called the Turing world within the Turing world, thought experiment that I did several episodes ago.</p>

<p><em><strong>[01:38:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Oh, yeah, that was good.</p>

<p><em><strong>[01:38:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:38:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Does this go against what we're talking about?</p>

<p><em><strong>[01:38:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It doesn't.</p>

<p><em><strong>[01:38:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:38:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I know it feels like it does.</p>

<p><em><strong>[01:38:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right?</p>

<p><em><strong>[01:38:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So you've got the, just to repeat the thought experiment, imagine you have Bob and AGI living inside of a virtual world that's running on a computer, and then you have another computer that's the exact same algorithm, and it's the same Bob running on the same computer, except it's running on a computer twice as fast as the other computer. So you watch Bob what he's going to do, and maybe Bob's in a whole society of AGI's, right? You watch what Bob's going to do, you look at even the knowledge he's going to create. And then you go and you say, Oh, I know Bob's about to discover, you know, quantum physics over on computer B, and computer A is slower and it's the same program. So I can predict that Bob is about to discover quantum physics on computer A because he's running at half the speed. And it's going to happen in October that he's going to invent quantum physics. And this just drives people nuts, right? Like they're like, oh my gosh, that shows that deterministic algorithms are predictable. And it shows that everything's wrong and there's no free will. So this must be wrong and people will argue with me over it. And they'll say, no, you're misunderstood. I had two crit rats that were vehemently arguing that no, actually the slower computer will end up doing something different. It's like, you realize that violates computational theory, right? You know, like, no, no, no, you're just not taking into consideration. They actually tried to invoke Deutsch's concept of fungibility in quantum physics to try to prove that something was wrong with this thought experiment.</p>

<p><em><strong>[01:40:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And they tried to invoke the idea that the computer would eventually make a mistake. And it was a really almost humorous argument because they just did not want to admit that you can, in this case, predict what's going to happen.</p>

<p><em><strong>[01:40:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Here's the thing, right?</p>

<p><em><strong>[01:40:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The reason why you can make this prediction is because you now are an observer outside these two realities, which is not the situation of an observer within a universe. And if you were to, let's say that, let's say that you were interacting, part of the reason why this whole thought experiment works is because one of the starting assumptions is that you're not interacting with these universes. Okay, let's say you were interacting with these universes. Okay, so you talk with both these bobs, the one that's running at half the speed and the one that's running at full speed. That interaction, so you tell, maybe you tell Bob A, you know what? Bob B just invented quantum physics, so just keep going, you're going to invent quantum physics.</p>

<p><em><strong>[01:41:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's just not the way it works.</p>

<p><em><strong>[01:41:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Okay, they're not, they're not going, they have different inputs now. They're not going to be predictable anymore. The fact that Bob B created quantum physics, now that he's interacting with that universe in some way, that whole computation now needs to be taken into consideration across them and including our real world. There's not even a reason to believe Bob B will invent quantum physics anymore because he's going to just live his life differently because the inputs are different. Okay, so they're back to being unpredictable now. The only reason, the only sense in which they are predictable is if you keep them completely isolated and you are an outside observer. It's not even hard to see why this must be the case. Because the moment I start to interact with the two Bobs, my computation is part of now the overall computation. Because whatever is going on in my world that you have to work out through computing the whole real world now drives some of the inputs when I have conversations with the two Bobs. And those conversations aren't going to be identical, right? So now everything in nature in the real world is impacting my mood when I'm talking to the two different Bobs and they interact. You know, once I know something about one Bob, I talk to the other one.</p>

<p><em><strong>[01:42:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> He's a different person.</p>

<p><em><strong>[01:42:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He diverges over time now because the whole computation has to now be taken into consideration, including the entire computation of the real world. Okay, so that's why you can't really, I mean, yes, it's predictable in the sense that you can predict it the second time, but isn't that kind of the same as saying it's not predictable? And this is where I think people will just sort of get really confused on this, right? Because the second Bob isn't predictable in the traditional sense, you're really just running the same algorithm twice. And yeah, of course, you know the outcome once you've run it.</p>

<p><em><strong>[01:43:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Of course you do.</p>

<p><em><strong>[01:43:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So now with that whole thing in mind and that whole additional explanation, let's talk through how this applies to Wolfram's theories. Okay, really the question we're asking is what if you just use a faster computer? So recall that any specific computation in the real world is, according to Wolfram, at maximum speed, or so he's conjecturing. If that conjecture is true, then we can conclude that computations on a computer are not at maximal speed, for the simple reason that PCs keep getting faster. So Rucker introduces the idea of what he calls strong unpredictability.</p>

<p><em><strong>[01:43:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I do not like that term.</p>

<p><em><strong>[01:43:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Probably should have been called strong intractability or something like that.</p>

<p><em><strong>[01:43:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay, but let's go with it.</p>

<p><em><strong>[01:43:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So on page 428, he defines it as, P is strongly unpredictable if and only if there is no Q that can emulate P and is faster than P.</p>

<p><em><strong>[01:43:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is the two Bob's problem.</p>

<p><em><strong>[01:44:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I shouldn't call it a problem.</p>

<p><em><strong>[01:44:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> The two Bob's thought experiment.</p>

<p><em><strong>[01:44:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> None of the computations on your PC are strongly unpredictable in that you can wait 18 months and then do the same computation faster. Interesting side note, we tend to think of Moore's law doubling every 18 months as an exponential speed up. And if the algorithm in question is polynomial, that is the case. But most algorithms are intractable due to being exponential on the inputs. So Moore's law actually only gives such algorithms a linear speed up, not an exponential one, because the problem is itself exponential. So, example, because a chess min-max search algorithm is itself exponential, you need quite a few exponential speed ups to go from, say, four plies or moves ahead to five. And so Moore's law doesn't create, doesn't make every algorithm exponentially speed up.</p>

<p><em><strong>[01:45:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That just isn't the case, okay?</p>

<p><em><strong>[01:45:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> In fact, the vast majority of algorithms are presumably NP. P is part of NP, but it's a small part of NP. You have to go back, if you don't know what I'm talking about, go back and listen to the computational theory episodes of the podcast. The net result of this is, is that a lot of things that we try to imagine in Hollywood, where we just imagine computers just getting faster and faster because of Moore's law, and therefore we've got this giant super intelligence that's 10,000 times smarter than us, and where we live inside this matrix that's richer than the real world or something like that, right? All of that is based on a misunderstanding of what happens with Moore's law.</p>

<p><em><strong>[01:45:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:45:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> This I really want to understand, and I'm not sure that I get what you're saying. So you're saying that the algorithms don't go any faster, even though the computers are becoming more and more faster?</p>

<p><em><strong>[01:46:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Let's use the chess example.</p>

<p><em><strong>[01:46:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:46:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So you're going to write a chess program, Peter.</p>

<p><em><strong>[01:46:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:46:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And you're going to make it run by trying to make as many moves ahead as it can. So it tries one move, then it tries doing the move of its opponent. Yeah. Then it tries doing the move that it's going to move against its opponent. Well, that's not going to work. It's going to have to try every combination.</p>

<p><em><strong>[01:46:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[01:46:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And then you get into the more atoms and the more chess moves on the board than there are atoms in the known universe kind of a thing.</p>

<p><em><strong>[01:46:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right.</p>

<p><em><strong>[01:46:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So when you do an exponential speed up, this is, by the way, I got this, like I should have looked this up, but there's a famous book, the algorithms book that they use in, you know, undergrad and grad programs. It was one of my textbooks for my computer science master's degree. And they have a whole page with an aside where they talk about this, where they say, it's actually really, it says, does Moore's Law compensate for algorithms being slow?</p>

<p><em><strong>[01:47:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And they said, no, it doesn't.</p>

<p><em><strong>[01:47:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And they pointed out that almost all algorithms were exponential on the inputs.</p>

<p><em><strong>[01:47:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:47:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:47:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So while the computer may be exponentially faster with Moore's Law doubling every 18 months or whatever, because the algorithms are themselves exponential on the inputs, you only get a linear speed up, right? They're both exponential, so they offset each other, right? So getting back to the chess program example, let's say your computer is fast enough that it can look ahead three moves.</p>

<p><em><strong>[01:47:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:47:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It can try every combination three moves out, which isn't that much, right? So let's say you say, you know, you're thinking Hollywood style. You know, computers in ten years, they're going to be exponentially faster. So that I predict that means I'm going to be able to look ahead a thousand moves.</p>

<p><em><strong>[01:48:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:48:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Because I'm going to have, you know, a computer thousand times faster, right?</p>

<p><em><strong>[01:48:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> In ten years.</p>

<p><em><strong>[01:48:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:48:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's a total misunderstanding. You will, after it's a thousand times faster, be able to go four moves ahead. Because you need a thousand times faster computer to go from three to four.</p>

<p><em><strong>[01:48:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> I see.</p>

<p><em><strong>[01:48:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I see.</p>

<p><em><strong>[01:48:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:48:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's interesting.</p>

<p><em><strong>[01:48:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So is nature usually maximally efficient in terms of speed like Wolfram believes?</p>

<p><em><strong>[01:48:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:48:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Surely there are some computations in nature that we can emulate faster than nature itself. So it can't be a universal truth. But now we need to get a bit more serious about if we're talking about Wolfram's version of emulation, which is a specific phenomena, or if we mean Deutsch's, which is not. I don't doubt we can emulate a waterfall to a level of accuracy that at some point, I can't tell the difference between a real waterfall and a fake one.</p>

<p><em><strong>[01:49:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That would be consistent with Deutsch's Church-Turing-Deutsch thesis claims that we can simulate anything.</p>

<p><em><strong>[01:49:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I do doubt that it would be possible to, down to the atom, emulate a specific real-life waterfall exactly on a computer. So if true, that really is equivalent to the claim that the waterfall is at maximal speed and thus strongly unpredictable. No amount of Moore's law can ever emulate that waterfall faster than the waterfall itself. Now, Rucker says, My sense is that the most complex physical processes are strongly unpredictable in that sense, that they represent computations that can't be run any faster at all.</p>

<p><em><strong>[01:49:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This is from page 430.</p>

<p><em><strong>[01:49:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This raises a difficult and to some troubling question. The question of if artificial super intelligences are physically possible. Now, you and I have talked about this in past podcasts. You raised it, and I made the statement, I don't actually know if you, like if we knew what the software was that a human brain was doing, and we ran it on a very, very, very fast computer today, would it be fast enough to run the software that the brain is running? There's an assumption that not only would it be fast enough, but it would be thousands of times faster, maybe hundreds of thousands of times faster. I don't think we have any reason to believe that's true. I don't know that it's false, right? Like I just, I think that those sorts of Hollywood guesses like that are entirely based on weird assumptions that don't actually make any sense. And we of course don't know how fast the brain actually is. Like we know how fast an individual neuron is, but it's massively parallel processed. And we don't even understand what it's doing, right? Or how it interacts physically and how those, like if you were to try to emulate everything in the brain that's going on, it would be a massive computation. You know, down to the microtubules to put Roger Penrose here, it would be a massive, it would be completely intractable for any computer to do. So there's kind of an assumption that we know enough about how the brain works, that we can assume it's a simple enough computation that we can say that the rest of what's going on is not necessary, right?</p>

<p><em><strong>[01:51:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That it's a level of emergence so we can ignore most of the physical processes. I just don't know that we even know that, right? Like we know so little about how the brain works on this that this is why Roger Penrose is trying to argue that actually part of our computation is done by quantum effects in the microtubules is because he can get away with that. I doubt that's true, but he can get away with that precisely because we have such a huge level of ignorance at this point.</p>

<p><em><strong>[01:52:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay, so I don't know what the answer is, but let's explore this a little.</p>

<p><em><strong>[01:52:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So the idea that nature is maximally fast would seem to suggest that artificial superintelligence as ASIs are impossible. But it isn't clear because the only absolute bar is that we can make a computation exactly equivalent to you that can emulate you with 100% accuracy.</p>

<p><em><strong>[01:52:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That is at maximal speed.</p>

<p><em><strong>[01:52:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So let's ask the question a different way.</p>

<p><em><strong>[01:52:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Would an artificial general intelligence automatically be an artificial superintelligence? It's tempting here to try to cite universal explainership and claim there is no such thing as an artificial superintelligence. Perhaps this is true, but a valid response might be, well, if it's 100,000 times faster, that is what we mean by artificial superintelligence.</p>

<p><em><strong>[01:53:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay, fine.</p>

<p><em><strong>[01:53:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But why does there seem to be an almost universal assumption that our current hardware is 100,000 times faster than the human brain? Do people just automatically assume that because we can compute, we can program a computer to do a calculation that is much, much faster than a human can consciously do? Okay, but the brain's processing power isn't determined by its conscious processing power.</p>

<p><em><strong>[01:53:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's determined by its unconscious processing power.</p>

<p><em><strong>[01:53:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So let's say you go play the game of catch with your boys, Peter. Go try to program a robot to play catch. This is a very complicated calculation that's going on, that is hard to get a robot to calculate fast enough in real time. But humans unconsciously do it easily all the time. Don't even get me started on dogs doing it.</p>

<p><em><strong>[01:54:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> They're like amazingly good at.</p>

<p><em><strong>[01:54:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Is it because we know how fast a CPU is compared to a single neuron? Okay, but that ignores the massively parallel processing the brain does. Okay, so now Rucker, page 254. The fact that our modern computer hardware is essentially serial tends to discourage us from thinking deeply enough about the truly parallel algorithms being used by living organisms, not just in the brain, but all throughout the organism. And using search methods to design the parallel algorithm takes prohibitively long. So until we know how to program an AGI, we sincerely have no idea, none, if the first AGI will be 100,000 times faster than us or one 100th the speed of us. Okay, like there's just no basis for deciding it's one or the other.</p>

<p><em><strong>[01:54:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[01:54:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Let me just use this interesting quote from Rucker. So in one of his science fiction stories, he imagined an experiment where they could speed grow a fetus. He admits that this is probably physically impossible and it was just a conceit needed for the story. So on page 433, he says, if you tried to speed grow a fetus, you'd likely end up with something that looked more like a stork or a cabbage than a baby. So this idea that reality is at maximal speed is kind of an interesting counterpoint to the very concept of an artificial super intelligence. Are you getting what I'm saying here, right? Like I can't say for sure that there couldn't be such a thing as an artificial super intelligence. That would require me to make some assumptions that don't quite follow from everything we're talking about. But it does seem like it puts some pretty, it forces you to really stop and think about this. I'm wrapping up here, but let me actually read a quote from, this is from off of Twitter, Deutsch Explains.</p>

<p><em><strong>[01:56:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So they try to quote Deutsch.</p>

<p><em><strong>[01:56:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Sometimes they quote him wrong.</p>

<p><em><strong>[01:56:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I don't know if this one's an exactly correct quote, but it's a long quote, so I think it's probably correct. 50 years ago, some far-sighted people realized that the new technology of a jetliner was soon going to make it normal for ordinary people to travel abroad for jobs and education and holidays.</p>

<p><em><strong>[01:56:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And they were right.</p>

<p><em><strong>[01:56:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> The revolution happened.</p>

<p><em><strong>[01:56:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But they also thought that by today, all that jet travel would be supersonic. And that has not happened. People at some point decided that supersonic travel is morally unacceptable. And also, the moon colony, the Mars expedition, nuclear power, by even the late 1970s, people no longer wanted what people had wanted at the early 60s. And because of that unpredictable change, their prediction of what our lives were going to be like was false. That's part of a wider impediment to prediction, namely unforeseen problems. The fads and fallacies and blunders that are going to seem like a good idea at the time in the next 50 years by definitions we can't foretell now. So predicting our future is nothing like predicting the strength of a bridge. Every significant innovation has unpredictable effects and they have knockoff effects. And after a few steps of that, the consequences and their consequences come to be the major component of what is happening. And as knowledge grows faster, the time for that to happen becomes shorter and shorter. The growth of knowledge is the only impediment to our ability to predict the future.</p>

<p><em><strong>[01:57:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> But in this respect, it's decisive.</p>

<p><em><strong>[01:57:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It will impose an ever closer planning horizon beyond which we are blind to the most important determinants of what is going to happen. So we face a paradox. The more we create knowledge, the less we know about our future. You know, I agree with everything in that statement except one sentence. And even that one, maybe I could read a terribly read as something I could agree with. The statement that I don't agree with is the growth of knowledge is the only impediment to our ability to predict the future. I do think the growth of knowledge is an example of what Wolfram's talking about. But I think that it is not the only example of what Wolfram's talking about. In fact, almost everything in nature qualifies, even if it's not creating knowledge. And because of that, things just are unpredictable. They kind of just are, right? And I suppose this is part of the reason why I just feel no fear over artificial superintelligences.</p>

<p><em><strong>[01:58:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And it's not...</p>

<p><em><strong>[01:58:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> When I say I feel no fear, I don't mean there is no danger. Of course, there's a danger. But this existential threat that some people feel, the AI doomers, it doesn't even make sense to me. I mean, doom is always around the corner.</p>

<p><em><strong>[01:59:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's surely true, right?</p>

<p><em><strong>[01:59:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I guess it could be by artificial superintelligence. It could be not by artificial superintelligence. It could be by artificial intelligences that are a different race, that are no smarter than us. And they just happen to beat us in a war and kill us all. Never mind if they're superintelligences or not.</p>

<p><em><strong>[01:59:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> There's all sorts of dangers.</p>

<p><em><strong>[01:59:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It could be that they're the only ones defending us from some crazy group of humans that want to wipe us out. I mean, it's who knows, right? I'm no more going to get worked up over the dangers of artificial superintelligence as real as that might be, then I'm going to get worked up of the danger of Nazis as real as that is, right? Like Nazis really were a threat, right?</p>

<p><em><strong>[01:59:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Yeah, no.</p>

<p><em><strong>[01:59:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And they are kind of a superintelligence.</p>

<p><em><strong>[01:59:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's hard not to read.</p>

<p><em><strong>[01:59:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> You know, when I read Nick Bostrom's book, Superintelligence, where he goes through all the different ways that AI or AGI could destroy humanity, it's hard not to get a little freaked out by that. But, you know, thinking it through more from the perspective you've presented here, it seems a lot less realistic. So, thank you. I'm even more optimistic than I was before, I guess.</p>

<p><em><strong>[02:00:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I think there are different ways to approach the AI-doomer thing. This is one of them, which is, look, you're making all sorts of assumptions that are just totally made up. They are, you know, getting back to the idea of rationality as severe testing, which is how I interpret Popper, right? Really, you're being irrational. It's not that you're necessarily wrong. It could be that the Nazis are going to kill us too. I'm just not going to spend any time worrying about that until it's an actual threat, right? Because I don't think it will be. I don't think we'll ever see the emergence of Nazis again.</p>

<p><em><strong>[02:00:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And they were a threat, though.</p>

<p><em><strong>[02:00:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Like they did emerge.</p>

<p><em><strong>[02:00:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> They were a threat, right?</p>

<p><em><strong>[02:00:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> A lot of the arguments that you hear don't make as much sense to me, like, oh, it would be racist, you know? Maybe, like, but our genes coerce us, and sometimes we don't mind it. Like, the fact that it makes sex pleasurable, and so we tend to think about it a lot and make sure romance is a big part of our lives and things like that. You know, I mean, like, that is the genes coercing us. Does that make the genes racist against us?</p>

<p><em><strong>[02:01:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> No, because we kind of like it, you know?</p>

<p><em><strong>[02:01:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> It's something, you know?</p>

<p><em><strong>[02:01:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I would argue that even moral feelings are things that are a form of kind of genetic coercion.</p>

<p><em><strong>[02:01:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> This one's going to be controversial.</p>

<p><em><strong>[02:01:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I'm talking moral feelings, not morality here, okay, just to make a distinction.</p>

<p><em><strong>[02:01:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That is, I think most of us are glad we have, right?</p>

<p><em><strong>[02:01:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Even though they're really kind of, from a certain point of view, the genes way of trying to manipulate us. And so I don't know, like maybe we will have AGI safety programs that may not be racist, that may be a good form of it. Like since we don't know what AGI's are, it's just really hard to even formulate the questions at this point. So I feel like most of the arguments, I think the answer is, look, it's just not a local problem, guys. Like it might be a problem in the future, but it's just not a local problem now. Anything you come up with to try to solve the problem, you know so little that you just don't even know what it is you need to address.</p>

<p><em><strong>[02:02:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> What about something like the paperclip factory, where you know about that Bostrom's thought experience, that experiment there where the computer wants to make a paperclip factory and then realizes that the most efficient way to do this is to destroy all of humanity and turn the world into a paperclip factory. I mean, is something like that a valid concern?</p>

<p><em><strong>[02:02:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, okay, so no, of course not. If the paperclip factory were smart enough to be able to overcome all of... So the assumption here is that it's not just a mechanical device, because if it's a mechanical device, then it's no threat at all, you just go turn it off, right? So we're trying to imagine a paperclip factory that also happens to be an artificial super intelligence. And in its desire to create paperclips, it sees the humans trying to shut it off as an impediment to its reward program. And so it decides to create a giant army of robots, because this is part of its goal to create paperclips. And it decides to wipe out the human race first, or something along those lines. Okay, to imagine this, you're trying to imagine this, you know, kind of savant, right?</p>

<p><em><strong>[02:03:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Yeah, well, it assumes that you could be a super intelligence and still be a psychopath, right?</p>

<p><em><strong>[02:03:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Which-</p>

<p><em><strong>[02:03:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Maybe you could be a super intelligence and a psychopath, but you would have to be a super intelligence and also not capable of understanding how to say, you know what, maybe my goal should be a different goal now.</p>

<p><em><strong>[02:04:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[02:04:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I don't even know how you would do that to a general intelligence.</p>

<p><em><strong>[02:04:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Or why not go into space and turn Jupiter into a paperclip factory?</p>

<p><em><strong>[02:04:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right.</p>

<p><em><strong>[02:04:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's starting with weird assumptions that just don't make sense, right? It's like you try to get explicit with it and immediately start going, but wait, it's a super intelligence. It can understand why the humans want to shut it off. It can understand why this is probably a bad idea to go to war with the humans.</p>

<p><em><strong>[02:04:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I would be more fearful of something like the great ooze [term wording uncertain].</p>

<p><em><strong>[02:04:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Are you familiar with that?</p>

<p><em><strong>[02:04:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I've not heard [wording uncertain].</p>

<p><em><strong>[02:04:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That was from Boston [name uncertain].</p>

<p><em><strong>[02:04:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Or maybe I heard it.</p>

<p><em><strong>[02:04:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> I don't remember.</p>

<p><em><strong>[02:04:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We make a bunch of really stupid little nanobots, and they're misprogrammed in some way, so they take every atom and they just break the atoms down, and you end up with just kind of dust, right? It's just this great ooze [term wording uncertain]. And so they start to do that to the entire world.</p>

<p><em><strong>[02:05:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> They're mindless.</p>

<p><em><strong>[02:05:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They're not artificial superintelligence, they're just little automatons. But they exist at the atomic level, because you can't see them. And they just basically, they just turn the whole world into this gray dust, and everybody's dead. That scares me more. I wouldn't say I'm exactly scared of that, because we're nowhere even close to having to worry about something like that. And by the time we are, we'll have the knowledge of how to deal with something like that. But that's scarier to me than an artificial super intelligence, where I can actually go in and talk to the thing, and say, you know what, this is kind of a bad idea.</p>

<p><em><strong>[02:05:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And reason with it, right?</p>

<p><em><strong>[02:05:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So because of that, it's not that there isn't potential danger with AGI.</p>

<p><em><strong>[02:05:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Of course, there is potential danger.</p>

<p><em><strong>[02:06:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And maybe there will even someday be something like a super intelligence in the limited sense that they can think faster than a human being. Like, that's not completely unthinkable that that could happen, right? I don't know if, I have severe doubts they could be a hundred times faster than us, right?</p>

<p><em><strong>[02:06:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Or a million times faster than us.</p>

<p><em><strong>[02:06:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I suspect that the human brain is so close to efficient in terms of its computation that we'll be lucky to get something that's like ten times faster.</p>

<p><em><strong>[02:06:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Where does memory come into this? I mean, it's not hard to imagine that they would be, have a million times more memory or something, right?</p>

<p><em><strong>[02:06:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> No, no, it isn't.</p>

<p><em><strong>[02:06:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You're right about that. So you would have to specify what we mean by memory. And this is something that's, we had an episode about this, the one where we did the Argue Me Anything episode. To a universal explainer, the concept of memory is unclear, right? So for a Turing machine, there's kind of this clear concept of memory, the tape. But there's no reason to limit the tape, because if you say, well, it's 100,000 long, then yes, there's certain algorithms you can't run, but you can always just lengthen the tape.</p>

<p><em><strong>[02:07:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> And I've got a phone right now that basically takes my memory into practically infinite.</p>

<p><em><strong>[02:07:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So to a universal explainer, the concept of memory, presumably when you asked the question, you were conceptually thinking remembering something in the brain, right? But that's not really your memory. Your memory is not really held in your brain. Like part of your memory is as a universal explainer, right? Like your phone is a part of your memory. Stuff you write down is a part of your memory, right? The textbooks you buy are a part of your memory. And so the concept of memory for a universal explainer doesn't equate easily to the concept of memory for a Turing machine. And I think the kind of the Deutsch answer here is probably pretty good. That in some sense, you're already augmenting your memory as a human. And there's no particular reason why we couldn't insert something into your brain and give you the ability to Elon Musk style, be connected to the cloud to store stuff. I don't know that a super intelligence, a so-called super intelligence would be any different than that, right? Like if you try to imagine it just a straight memory, it's going to have to access it just like we do. If you're trying to imagine it more like the memory of the brain, where it's kind of contained within the neurons, then you have to keep imagining a larger and larger brain to hold this that still have the connectivity of neurons.</p>

<p><em><strong>[02:08:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> And that's going to slow the whole process down, right?</p>

<p><em><strong>[02:08:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So it's actually going to start being slower.</p>

<p><em><strong>[02:08:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So at some point, you almost have to say, look, I'm going to let the certain, the universal explainership be held within something that's reasonably sized, that runs at a decent speed, and then I'm going to have to just attach memory. And now it's almost exactly like what a human's doing when they augment their memory with their phone, right? So I don't know that, it doesn't seem to me that really we have to even worry about the possibility of an artificial superintelligence, but I will admit that that's a little bit subjective on my part. There could be such a thing as an artificial superintelligence, and we just don't know. Like it's really we just don't know. It seems like the impediments are much larger than people realize, but they could be addressed maybe.</p>

<p><em><strong>[02:09:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Like maybe we could make something a hundred thousand times faster than us.</p>

<p><em><strong>[02:09:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I can't say we can't, but the impediments are way higher than people think.</p>

<p><em><strong>[02:09:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So at least.</p>

<p><em><strong>[02:09:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Okay, I wasn't planning to do this, but let's go ahead and let's talk about this kind of AI doomer all the way. When you are, what's the guy who's the big AI doomer that really popularized it?</p>

<p><em><strong>[02:09:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Eli—Eliza [name recall uncertain], what's the [wording uncertain]?</p>

<p><em><strong>[02:10:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Oh, oh, oh, this guy, yeah, Eliza [name pronunciation], Yud—Yudkowsky.</p>

<p><em><strong>[02:10:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yeah, and that's the easiest name to pronounce, Eliza.</p>

<p><em><strong>[02:10:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> He's got a very long book.</p>

<p><em><strong>[02:10:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> That's right.</p>

<p><em><strong>[02:10:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[02:10:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah.</p>

<p><em><strong>[02:10:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Okay.</p>

<p><em><strong>[02:10:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, and he's also one of the most important people in terms of popularizing—</p>

<p><em><strong>[02:10:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Bayesianism.</p>

<p><em><strong>[02:10:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Yeah, yeah.</p>

<p><em><strong>[02:10:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Bayesianism.</p>

<p><em><strong>[02:10:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> [wording uncertain].</p>

<p><em><strong>[02:10:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So, so, you know, crit rats hate him for both reasons. You know, when you're taking his point of view, right, if you're trying to put yourself into his mindset, the things he's imagining, if you were to make them explicit, they sound as silly as they actually are, right? You're imagining that there's no theory of intelligence first, so that when you create your first AGI, you know you're creating an AGI. That you almost happen upon it by accident, by working with AI, and suddenly it's super intelligent. You didn't even realize that you just crossed the boundary into AGI-ness, right? And you have to imagine that it just so happens that the algorithm can be run 100,000 times faster on a modern computer. This person is probably not working on a supercomputer, so on a modern laptop, it's 100,000 times faster than its creator, right?</p>

<p><em><strong>[02:11:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> So it's a super intelligence.</p>

<p><em><strong>[02:11:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And you don't know, it doesn't go through childhood, right? It doesn't spend 30 years going to school to learn stuff before it's competitive with the human, you know, and can even take care of itself. It just sort of happens to be that all the knowledge is right there, the moment it's born, and it's already smarter than you in terms of, you know, knowledge that you have. And it just so happens that it's psychopathic and it has no morals. And like the level of things he's assuming that if you were to call them out explicitly, you would immediately go, wait, that doesn't make sense, right? There's tons of them going in there, where he's just making all sorts of assumptions that you go, wouldn't it make more sense that we would have to have a theory of intelligence first before we can make an AGI, as Deutsch has, in my opinion, correctly argued. It's not that there is zero chance we might stumble upon it by accident.</p>

<p><em><strong>[02:12:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> In a certain sense, evolution did, right?</p>

<p><em><strong>[02:12:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I don't know if you can really call it by accident, because evolution is in some ways very purposeful. But again, controversial statement. But the fact is, though, is that we probably aren't going to. It just isn't the way these things work. You don't have no idea what you... Even in machine learning, where we often stumble upon things we don't understand, the odds that we would just happen to stumble upon AGI-ness with zero understanding of what we're doing in advance, right? I don't even understand where that thought's coming from, right? So I could almost believe it if you were trying to emulate the brain neuron by neuron, like the blue brain project, I think that's what it's called.</p>

<p><em><strong>[02:13:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Right?</p>

<p><em><strong>[02:13:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> But even then, is it going to be slower than us?</p>

<p><em><strong>[02:13:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's really unclear why they're so scared, right? Because the assumptions that they're adding up, that they're tacitly assuming they're, all of them seem really questionable. So it seems way more likely that we'll evolve with them, right? That once we start, once we have a theory, we'll say, look, let's make our first AGI, and then they'll be like children, and we'll have to like talk with them, and we'll understand how they interact. And they won't be super clever right out of the bat, because that's something you have to learn over time. They'll probably be slow, and painfully slow compared to us, initially. It may take them like 50 or 100 years to get to where we are by age 20, you know, or something like that. Right. There are so many assumptions going in here that could just easily go the other way, and in fact, there’s obstacles that would even almost suggest they should go the other way, unless you can explain to me how we overcome the fact that the brain is doing this massive parallel computation and [wording uncertain] our computers are serial, for example, right? Someone’s going to have to probably build a massively parallel architecture to be able to get a brain to work as fast—the artificial brain to work as fast as a real brain would be, my guess, right? So we’ll do it once we have a theory of how to do it. Someone will fund that and they’ll build some sort of massively parallel specialized computer to be able to do it, but it’s not going to, like, escape onto the internet and run on a laptop somewhere, you know, as Hollywood style. It's just this—</p>

<p><em><strong>[02:15:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Like I'm trying to say here is the doomer ism. It's not that there isn't potential doom. That's not what I'm trying to say. It's that the reason why they're hyper focusing on this one thing is because they have really, really, really, really questionable assumptions, [wording uncertain], right?</p>

<p><em><strong>[02:15:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> Well, we should probably wrap this up, but I find your take on this completely compelling. This was a great episode. I really learned a lot from listening to you today.</p>

<p><em><strong>[02:15:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> All right.</p>

<p><em><strong>[02:15:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Thank you.</p>

<p><em><strong>[02:15:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Hello again.</p>

<p><em><strong>[02:15:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#2563eb">Peter:</span></strong> If you've made it this far, please consider giving us a nice rating of whatever platform you use, or even making a financial contribution through the link provided in the show notes. As you probably know, we are a podcast loosely tied together by the Popper–Deutsch theory of knowledge. We believe David Deutsch's Four Strands tie everything together, so we discuss science, knowledge, computation, politics, art, and especially the search for artificial general intelligence. Also, please consider connecting with Bruce on X at bneilson01 [handle spelling uncertain]. Also, please consider joining the Facebook group, The Many Worlds of David Deutsch, where Bruce and I first started connecting.</p>

<p><em><strong>[02:16:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unidentified speaker:</span></strong> Thank you.</p>
