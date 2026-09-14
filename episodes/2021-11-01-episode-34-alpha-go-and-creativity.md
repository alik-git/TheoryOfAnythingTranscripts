---
layout: default
title: "Episode 34: Alpha Go and Creativity"
parent: Episodes
nav_order: 34
permalink: /episodes/34/
---

# Episode 34: Alpha Go and Creativity

- Links to this episode: [Spotify](https://podcasters.spotify.com/pod/show/four-strands/episodes/Episode-34-Alpha-Go-and-Creativity-e16tnnb) / [Apple Podcasts](https://podcasts.apple.com/us/podcast/episode-34-alpha-go-and-creativity/id1503194218?i=1000540352208&uo=4)

> Unofficial, reviewed transcript. It may still contain mistakes; check the podcast when wording matters.

Speakers are labeled by first name where identified. Unidentified-speaker labels may refer to different people. Bracketed question marks indicate uncertain wording.

## Transcript

<p><em><strong>[00:00:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Welcome to The Theory of Anything Podcast. Hey guys, how are you doing today?</p>

<p><em><strong>[00:00:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Doing great, Bruce.</p>

<p><em><strong>[00:00:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> How are you?</p>

<p><em><strong>[00:00:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Good.</p>

<p><em><strong>[00:00:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Well, mostly I'm, I am actually doing well. You probably know I have a kidney stone, so I'm not necessarily doing great.</p>

<p><em><strong>[00:00:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> How are you, Tracy?</p>

<p><em><strong>[00:00:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> How are you, Tracy?</p>

<p><em><strong>[00:00:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> I'm well.</p>

<p><em><strong>[00:00:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Good.</p>

<p><em><strong>[00:00:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Today, we're going to do Alpha Go the Movie, which is a movie that is free on YouTube. Just go Google Alpha Go the Movie. And don't listen to this episode until you've gone and you've watched that movie.</p>

<p><em><strong>[00:00:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This episode is going to contain spoilers. Now, you may think, no, wait, isn't Alpha Go the Movie a documentary? How can a documentary have something that can be spoiled?</p>

<p><em><strong>[00:00:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'm telling you, go watch the movie, and you don't want to listen to this episode first because it's going to contain spoilers. I thought the movie was just excellent. It was tense, it was exciting.</p>

<p><em><strong>[00:01:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I had been asked to watch it for one of my classes. I think it was my deep learning class. They required it as part of the curriculum, required in the sense that nothing in college is truly required ever.</p>

<p><em><strong>[00:01:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But it was something we were supposed to do. So I went and watched it, and I was so excited by the end of the movie. And you know what?</p>

<p><em><strong>[00:01:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I knew what was going to happen. I was already familiar with the story. So even though, so it had already been spoiled for me to some degree.</p>

<p><em><strong>[00:01:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I was so excited about it. I took my wife and whichever kids I could get to watch it with me. And I played it on the in the theater downstairs.</p>

<p><em><strong>[00:01:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And my wife was like so tense all during the movie. It's just an excellent dramatic story that happens to actually be true. Sorry.</p>

<p><em><strong>[00:01:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Go ahead, Tracy.</p>

<p><em><strong>[00:01:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> I was just saying it was very suspenseful.</p>

<p><em><strong>[00:02:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yes, it is. And we're going to spoil it all. So go watch the movie first and then come back to this episode.</p>

<p><em><strong>[00:02:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And we're going to do kind of an analysis of the movie. And I've already kind of spoiled parts of it. Like if you watched our reinforcement learning episode, we talked quite a bit about Alpha Go the Movie in there.</p>

<p><em><strong>[00:02:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I probably should have warned people then that there was spoilers in it. Anyhow, let's just start at the beginning here with the story. So David Silver is a famous guy in machine learning, in reinforcement learning.</p>

<p><em><strong>[00:02:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And he has a series of lectures that are available on YouTube, which are excellent, where he teaches reinforcement learning. So if you watched our reinforcement learning episode and it made you curious, you can learn about the full theory of reinforcement learning at a college level. From David Silver, probably one of the world experts in reinforcement learning.</p>

<p><em><strong>[00:02:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He's a very good teacher. I say the only real downside to the lectures is that I had a hard time understanding him sometimes. He has an accent and the sound isn't the greatest.</p>

<p><em><strong>[00:03:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But he's someone who really knows this material well. He is in charge of Alpha Go. So the setup for this is that Go is sometimes called Chinese chess, which is a total misinterpretation.</p>

<p><em><strong>[00:03:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But it's kind of their strategy game that they like, similar to how we in America might enjoy chess. But the games are nothing similar at all. It's a much harder game to write a program for to play.</p>

<p><em><strong>[00:03:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And the reason why is because the way they usually do game playing algorithms is with the minimax algorithm, which basically just tries making a move, tries making a move for its opponent, tries making its own move. It just tries as many moves as it can out into advance as far as it can. Quickly, that becomes an exponential nightmare.</p>

<p><em><strong>[00:03:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You know, if you can get seven moves out, you're doing great. With Go, the branching factor, the number of possible moves is so large, you just can't really use the minimax algorithm effectively. And then to make matters worse, it's not—in chess, you can kind of come up with really simple algorithms that can tell you, oh, you know, your seven moves out, if you do this move, seven moves from now, you'll be in a better position.</p>

<p><em><strong>[00:04:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And it can determine, quote, better position based on some simple algorithm that says, well, you know, you haven't lost your queen, and you've got this many points for the pieces on your board. And they can come up with very simple algorithms that tell you if the board position, seven moves out, is good or not. There's nothing equivalent for that, for Go.</p>

<p><em><strong>[00:04:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Go masters use their intuition to be able to tell if their board position from this move is good or not. And so how do you get a computer to do intuition? So this is kind of the setup that there are Go playing algorithms when David Silver steps onto the scene and they're really bad.</p>

<p><em><strong>[00:04:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They can only play at an amateur level. Professionals think it's a joke. It's kind of a common joke amongst professional Go players about how bad Go playing computer programs are.</p>

<p><em><strong>[00:05:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> OK, so this is kind of the background. They don't, I don't know if they fully explain that in the movie or not, but that is the background for which David Silver then decides, I'm going to make this Go playing algorithm that can actually compete at a professional level and gets a team together as, you know, for his university or whatever, whoever is funding it. And that is what AlphaGo is.</p>

<p><em><strong>[00:05:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They're setting out to try to beat professional level players. If possible, they want to beat the world champion who's Lee Sedol. They don't know what it's going to take to do it.</p>

<p><em><strong>[00:05:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They're just going to try to use the theory of reinforcement learning. They feel like that and deep reinforcement learning—I'll explain what that is in just a second—are technologies that in principle might be possible to build a professional level go playing algorithm out of.</p>

<p><em><strong>[00:05:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But no one in the world has ever done it before. They don't know if it can really be done or not. And then here's a quote from David Silver.</p>

<p><em><strong>[00:06:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And we've often talked about the fact that so much of what goes into machine learning is human knowledge. You know, machine learning does create knowledge, but it's not very much. It's mostly knowledge coming from humans.</p>

<p><em><strong>[00:06:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And David Silver admits this. This is a quote from the movie. He says, everything that AlphaGo does, it does because a human has either created the data that it learns from, created the learning algorithm that learns from the data, created the search algorithm.</p>

<p><em><strong>[00:06:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> All of these things have come from humans. So really, this is a human endeavor. And this is actually an important point because everybody is kind of cheering on the human and not the computer because we relate to Lee Sedol being human.</p>

<p><em><strong>[00:06:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But he's kind of making the point both sides are, in a sense, human, right? That you've got this team of programmers that are all human and they want to be able to make an algorithm that has never existed before, that can do something that's never been done before. This is a very human thing for them as well, right?</p>

<p><em><strong>[00:07:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That they want to be successful. So for this movie, this made this somewhat exciting for me. You can't help but cheer for Lee Sedol, the human player.</p>

<p><em><strong>[00:07:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But you're not really against Team Alpha Go, because they're a bunch of humans that you really kind of are rooting for too. At least that was for me.</p>

<p><em><strong>[00:07:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What did you guys think?</p>

<p><em><strong>[00:07:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Everybody is a very sympathetic character. You end up rooting for everybody, and it's hard when people are getting beat that you know that they've dedicated their entire life to being masters, and now they're getting beat by a computer that they don't even understand.</p>

<p><em><strong>[00:07:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> So I found everybody to be a sympathetic character.</p>

<p><em><strong>[00:07:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yeah. With this in mind as the background, Team Alpha Go contacts, I can't even pronounce his name, but contacts a Go player who's a professional level, but not a strong professional level player. He's kind of somewhere in the middle somewhere.</p>

<p><em><strong>[00:08:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They contact him and they say, we're working on a program to play Go. We need someone who's good at Go to help us. We'd like to pay you to come in and be part of our team.</p>

<p><em><strong>[00:08:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And he's thinking, okay, this is dumb. You know, computers can't play Go. I'll show up.</p>

<p><em><strong>[00:08:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'll see what they're into. I don't know what they need from me. You know, and he's not really sold at all on what's going on when he first shows up.</p>

<p><em><strong>[00:08:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What happens when he plays his first game against him? Do you guys remember?</p>

<p><em><strong>[00:08:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> I just watched it this morning. I think he loses.</p>

<p><em><strong>[00:08:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He does. He loses his first game. And he's so embarrassed that he walks out.</p>

<p><em><strong>[00:08:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He walks out. And the team AlphaGo is thinking, he might not come back. He wanders off and he's thinking to himself and he's just humiliated because to lose to a computer in Go is like the most humiliating thing that could ever happen to you as a Go player because it's so well known that Go programs just don't play well.</p>

<p><em><strong>[00:09:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And he's walking along, being all humiliated, it finally strikes him. Oh my gosh, they've got something. It just beat me.</p>

<p><em><strong>[00:09:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And he runs back and he goes, I've got to be a part of this.</p>

<p><em><strong>[00:09:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Once he can put his own ego aside to realize what a phenomenal breakthrough it is.</p>

<p><em><strong>[00:09:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yes. He's like, they've made a Go playing program. So he runs back and he wants to participate.</p>

<p><em><strong>[00:09:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He gets super excited. And he's one of the main narrators of the documentary through the rest of the movie. And he's interesting because he's on Team Alpha Go.</p>

<p><em><strong>[00:09:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And you can tell he is. He really wants to see Alpha Go win from this point forward. But he's coming at it from the standpoint of the person that helped train Alpha Go.</p>

<p><em><strong>[00:10:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You know, that this is his baby to some degree. He's not at Lee Sedol's level. And he knows he's not, right?</p>

<p><em><strong>[00:10:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So Alpha Go is way going to exceed his skill level. And he knows this. But he kind of just understands what it's like to play Alpha Go and to be shocked when you suddenly realize this is no normal Go-playing program.</p>

<p><em><strong>[00:10:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Right. And so you get a lot of kind of the sense of what it's like to be a go player through his eyes throughout the movie. So they train up and they do a challenge to Lee Sedol, who's the world champion.</p>

<p><em><strong>[00:10:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Lee Sedol accepts and they're going to do five games. Lee Sedol, initially, he takes the stance, oh, I'm going to beat it all five games. And he seems pretty confident because, of course, he knows Alpha Go programs, they're bad.</p>

<p><em><strong>[00:11:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But at least I don't know about you guys, but it seemed to me like maybe he was just a little bit nervous. Like, you can see him kind of asking questions. Oh, I saw it playing your Go consultant.</p>

<p><em><strong>[00:11:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And has it gotten better since then? They're like, well, we can't tell you how much better it is. But you can kind of tell that he knows he's jumping into the unknown here a little bit.</p>

<p><em><strong>[00:11:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I thought that was interesting, too. Very human sort of thing. He starts off very confident, but he's not quite sure what he's dealing with.</p>

<p><em><strong>[00:11:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He's never played this program before. He's seen games with this other Go player who's not on his level. He thinks he can beat it based on seeing those games that it's played.</p>

<p><em><strong>[00:11:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But of course, they're improving the algorithm right up until the day of the game. So what he's actually going to be playing is not the same algorithm that he's actually seen. It's going to be something better by the time it gets there.</p>

<p><em><strong>[00:12:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I thought that was interesting, too. There's kind of this tension about how well can the algorithm play. And you get a good feel from Team Alpha Go, how nervous they are, that they're taking their little program that's only played a mid-level go player up to this point, and it's suddenly going to be playing the world champion, and it might embarrass them, right?</p>

<p><em><strong>[00:12:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Yeah, I think you can feel like this definite, like they're afraid that their program is going to get beat, you know, three moves in, four moves in.</p>

<p><em><strong>[00:12:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yeah. Yeah, and they're also worried about the game hallucinating. So that's what they call it.</p>

<p><em><strong>[00:12:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They're worried that the game will, at times, it will think it's winning when it's not. It's like hallucinating a victory that's not there. The game has a history of doing that, where, you know, some percentage of the time, it suddenly, its algorithm says, oh, you have, you know, a 90% chance of winning with this move.</p>

<p><em><strong>[00:13:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And it's just a bad move, right? It's completely off with its calculations. And when that happens, the game doesn't play like a human and still put up a decent fight.</p>

<p><em><strong>[00:13:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It just starts to do stupid things, embarrassingly stupid things. So the biggest fear is that it will hallucinate in the middle of the game and it will start to do really embarrassing moves. And when it loses, it won't lose, you know, gracefully or put up a good fight.</p>

<p><em><strong>[00:13:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It'll basically just start doing incredibly stupid things and the team will be embarrassed. So there's this real threat that the team might get embarrassed by Lee Sedol, who's the world champion Go player. So you kind of feel for them too.</p>

<p><em><strong>[00:13:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Tracy, I know you wanted to ask about hallucination. I tried to explain it there. Do you have any other questions about what it means to hallucinate?</p>

<p><em><strong>[00:14:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> Well, no, I was just confused by the term because I think most people think that you're literally imagining something that's not there. And in this case, for a computer, I guess I'm just thinking that… is ‘hallucinate’ the right word? It's kind of weird because I think the computer's—I don't know—is it imagining versus it's trying to project or predict versus just imagining something that's not there?</p>

<p><em><strong>[00:14:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yeah, you know, that's the word that I believe the team just used or maybe they used the word deluded. I think maybe they used both. But if you really think about it, this is just humans making up a word for something, right?</p>

<p><em><strong>[00:14:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I mean, it's—they call it deluded or hallucinate, but it's not—because it's somewhat analogous to when a human is deluded or hallucinates, but it's not really the same thing.</p>

<p><em><strong>[00:14:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's just an analogy to it.</p>

<p><em><strong>[00:14:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> So I will admit I was a little confused on how that would impact the programs. Like what did they think that the program would do once it started hallucinating that would be different from what they had been training it to do?</p>

<p><em><strong>[00:15:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So if it gets into one of these states, so to understand why this would happen, think back to our reinforcement learning episode. Okay. Now, even if you only just got the gist of the theory of reinforcement learning, remember that there was a world space, right?</p>

<p><em><strong>[00:15:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> There's this solution space where you have to have every have to have a table, a Q table, with every single possible combination that's possible in this world space. So what's the world space for AlphaGo? It's every possible configuration of the board.</p>

<p><em><strong>[00:15:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So imagine try to make an array and the array, the first element in the array represents a board with no pieces on it. And then the second element in the array represents a board with only white having one piece, you know, in the bottom right corner or something like that. And then you'd have to just have one something in that array for every single possible board position that could possibly exist according to the rules.</p>

<p><em><strong>[00:16:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So an array that large would be too large for any computer. There's no computer on the earth that could store that much memory. It's just too large.</p>

<p><em><strong>[00:16:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Right. And this is one of the problems with the theory of reinforcement learning, is that even for really small, the world spaces that it can actually handle, it can only handle small ones because this Q table grows out of control really fast. Right.</p>

<p><em><strong>[00:16:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's exponential growth. So, and then even if you could fit it all into memory, you've still got a second problem. The way it works, think about how when the little robot went up into the corner, slowly the numbers move out from the goal and the reward moves forward and it calculates a value for each space until it finds the spaces that are next to the goal.</p>

<p><em><strong>[00:17:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And then those get a higher value state. Okay. Remember how that works from the episode of Reinforcement Learning?</p>

<p><em><strong>[00:17:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Yes. Yes.</p>

<p><em><strong>[00:17:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So if you have that many spaces, how many games would you have to play for it to actually learn that this specific board configuration is this close to the goal? You would have to play probably trillions to the trillions games, right? I mean, it would be an enormous amount of games you would have to play.</p>

<p><em><strong>[00:17:35]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And this is the second problem with Reinforcement Learning, is that even if you can fit, and I actually tried this, I did a lunar lander reinforcement learning algorithm, and I tried to just make a state space that just used a regular Q table, and it crashed that lander like you wouldn't believe. The problem was, is that I just couldn't get it to train enough to actually fill up the table, so that most of the table was always zero, and it just didn't know what to do. Okay.</p>

<p><em><strong>[00:18:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So let's use this as kind of a starting point. Imagine that they actually implemented this with the Q table, which they didn't. I'll explain what they did instead.</p>

<p><em><strong>[00:18:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But it's a similar problem. Imagine that you got the game into a state that it has never seen before. So in its Q table, it gives it a value of zero.</p>

<p><em><strong>[00:18:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's going to basically make a random move at that point. So humans never make random moves. If it starts making random moves, it's going to look stupid.</p>

<p><em><strong>[00:18:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's going to look like it's a little child playing the world champion. Right. So it's going to be embarrassing for the programmers.</p>

<p><em><strong>[00:18:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Does this kind of make sense?</p>

<p><em><strong>[00:18:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Yes, it does.</p>

<p><em><strong>[00:18:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So since you can't possibly fit the world space for Go into a computer, and if you did, it wouldn't work anyhow because you'd never be able to play enough games, what do they do instead?</p>

<p><em><strong>[00:19:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What they do is they keep in mind that neural networks, they do a very good job of mimicking any function. So think of it like this. We somehow recognize faces.</p>

<p><em><strong>[00:19:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We go on Facebook, we see a picture, we recognize a person's face. So we know there's a function that humans use to recognize faces. We just don't know what it is.</p>

<p><em><strong>[00:19:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I talked about this in the machine learning episodes where visual recognition is something we don't really understand, and so we didn't know how to program it, and so it turned out to be easier just to let machine learning learn how to do it. Well, the way they do that is with neural nets. Neural nets will come up with some sort of function that works, at least well enough, right?</p>

<p><em><strong>[00:19:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And it doesn't necessarily do it the way a human does, and that's why they have these things called adversarial examples, where you have a picture of a dog, and it recognizes the dog, and then you go change a couple pixels, and it still looks exactly like a dog, and the human can't tell the difference, and now the computer thinks it's a giraffe, right? I mean, there's those funny things you can do because of the way it comes up with its functions, okay? But that's what neural nets do.</p>

<p><em><strong>[00:20:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Neural nets will take any function, and the whole world is made up of functions. That's part of the theory of computational theory, that everything can be simulated as a function. So we're basically just asking the neural net through its training algorithm, through using gradient descent as its heuristic.</p>

<p><em><strong>[00:20:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We're asking it to try variants of weights and select the best ones, which is what we talked about when we talked about knowledge creation and machine learning, until it finds some sort of local minima that works pretty well. You can do that for a Q table. You can say, okay, we can't fit the true Q table and solution space in there, so we're going to use a neural net that takes inputs like a Q table would and gives outputs like a Q table would, but it's just the neural net doing it instead, which is going to be much smaller.</p>

<p><em><strong>[00:21:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And neural nets are good at generalization. So the neural net will figure out on its own that this board position is somewhat analogous to that board position, and it will end up with some sort of knowledge about what to do for every single board position, even though you never actually reached some of the board positions that it has to play. But if the neural net hasn't been trained well, it will start to act a lot like a Q table that's reached a state where it just doesn't know what to do.</p>

<p><em><strong>[00:21:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you hit some combination and it thinks it's analogous to something that it isn't, or it just has never really seen anything like that before, it'll just start to do random moves, basically. It's the same problem, but on a much smaller scale. And I think part of the reason why AlphaGo had this particular problem is that the original AlphaGo used human training examples.</p>

<p><em><strong>[00:21:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It would train off of human games. The later AlphaGo didn't. It just played itself.</p>

<p><em><strong>[00:21:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So it didn't have to use any input of human games at all for its training. That version where it just played itself, it would just create its own data by playing games with itself and play it billions and billions of times. That version didn't seem to have anywhere near the level of the hallucination problems that the earlier problem did when it was still trying to use these human games, which means that, I mean, how many human games have been played in the history of the world?</p>

<p><em><strong>[00:22:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Right. I mean, AlphaGo, the final version that doesn't use human human games, it plays more games than have probably been played in the history of the world by humans. When you're trying to train off of the human knowledge, you're leaving all sorts of gaps in its knowledge as to what to do.</p>

<p><em><strong>[00:22:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Okay. But kind of a shortcut initially that you can show what a good game looks like, and you can use that for generating board positions, but it's going to be a weakness also. So that's really kind of what happened there from a technical standpoint, and they hadn't yet figured out how to get it to stop doing that.</p>

<p><em><strong>[00:22:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And so there was always this threat that the game might just embarrass them. Does that answer your question, Cameo?</p>

<p><em><strong>[00:23:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> It does. It does. That was a great explanation.</p>

<p><em><strong>[00:23:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So we get into the game, and do you guys remember what happened? Does one of you want to kind of summarize what happens in the first game there?</p>

<p><em><strong>[00:23:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> Well, it was just very tense. You know, there's no obvious face to an opponent, which is terrifying. And it started out really, to me, it seemed hesitant.</p>

<p><em><strong>[00:23:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> And it was anxiety-ridden because the Alpha Go took so long to make its first move. There was a lot of doubt. And like, is this really going to work?</p>

<p><em><strong>[00:23:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> How's this going to play?</p>

<p><em><strong>[00:23:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So what you just mentioned is really interesting. When humans play, there's only certain opening moves you can make. So humans make an immediate open move, right?</p>

<p><em><strong>[00:23:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They don't stop and think about it for 10 minutes. So Alpha Go stopped and thought about it for 10 minutes, about which move it wanted to make. And for an opening move, that seems like, wow, is it just confused?</p>

<p><em><strong>[00:24:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But it turns out it stopped and thought about each move for 10 minutes pretty much the whole way through. You know, it just always just stopped and thought for some period of time. And it was very unhuman like in that regard.</p>

<p><em><strong>[00:24:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So it kind of threw the human commentators and Lee Sedol off because they weren't quite sure what to make of that.</p>

<p><em><strong>[00:24:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Well, I think everybody is kind of thinking that the machine's not able to make the next move, right? Like they're they're not sure if because why would a machine need to think?</p>

<p><em><strong>[00:24:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Right.</p>

<p><em><strong>[00:24:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> You know, and and even that that phraseology like what what is the machine actually doing during these things? Is it running multiple variations of the move?</p>

<p><em><strong>[00:24:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yeah, it's it's running. It's probably the way AlphaGo is programmed. It does use a minimax algorithm.</p>

<p><em><strong>[00:24:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So it actually runs through every possible move, several moves out. It just does that regardless of where it is. It doesn't care if it's the first move or the tenth move or the hundredth move.</p>

<p><em><strong>[00:25:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It always would just try to run multiple different moves and try to think outward what its best move is.</p>

<p><em><strong>[00:25:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Yeah, but they they do it. All of the humans are like, why is it taking it so long?</p>

<p><em><strong>[00:25:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It kind of makes sense, though. A human player would have a heuristic in mind. They would think, well, there's only certain good opening moves.</p>

<p><em><strong>[00:25:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'm just going to pick one of them. Whereas the computer doesn't have a library of opening moves. They probably could have given it a library of open moves to make it more human like.</p>

<p><em><strong>[00:25:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But they didn't. Instead, the computer had to go run a simulation of a little game in its head to figure out what a good opening move would be. Does that make sense?</p>

<p><em><strong>[00:25:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Yeah, it does.</p>

<p><em><strong>[00:25:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So the first big surprise was what they call Move 37. Oh, another thing, to Tracy's point, Alpha Go is played by one of the programmers.</p>

<p><em><strong>[00:26:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So the programmer has a computer screen up. He enters the move that Lee Sedol makes, and then he waits for Alpha Go to come back. And when Alpha Go makes the move, he then does that move on the board, according to what Alpha Go told him to do.</p>

<p><em><strong>[00:26:16]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And Lee Sedol keeps trying to read the programmer's face.</p>

<p><em><strong>[00:26:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> Yeah, yeah, I remember, yeah.</p>

<p><em><strong>[00:26:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> It's disturbing.</p>

<p><em><strong>[00:26:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He's trying to figure out what the programmer's thinking. And then he suddenly realizes the programmer doesn't know what's going on.</p>

<p><em><strong>[00:26:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'm not playing the programmer, I'm playing the computer.</p>

<p><em><strong>[00:26:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Well, and it's especially funny because the programmer doesn't actually know how to play Go at all.</p>

<p><em><strong>[00:26:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He doesn't, yeah.</p>

<p><em><strong>[00:26:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> He doesn't understand the game at all.</p>

<p><em><strong>[00:26:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So a real player would read the tension in their in their opponent's eyes and would determine what's this person thinking and what they're doing. And Lee Sedol suddenly finds that that part of his skill set is useless because the person he's playing against sitting across from him on the table, it doesn't know anything. It totally can't give anything away.</p>

<p><em><strong>[00:27:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I thought that was an interesting thing there. So Move 37 happened while Lee Sedol was on a break. So Lee Sedol went up and went to go out and take a break.</p>

<p><em><strong>[00:27:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And the programmer who I believe they said his name is Aja. So here's what the Go consultant said. He says, so Aja, who's the human playing the Alpha Go, sees Alpha Go, plays Move 37, and Aja puts the stone on the board.</p>

<p><em><strong>[00:27:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> When I see this move, for me, it's a big shock. Normally humans, we never play this one because it's bad. It's just bad.</p>

<p><em><strong>[00:27:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And here's what the commentator said about Move 37. Oh, it's totally an unthinkable move. Yeah, that's a very surprising move.</p>

<p><em><strong>[00:27:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And then one of the programmers said, I thought it was a mistake, and he laughed. So Move 37 gets made, and the initial impression of these knowledgeable commentators, and the Go consultant, who's a knowledgeable mid-level Go player himself, is, uh-oh, maybe that's a mistake. Lee Sedol comes in, and the Go consultant's waiting for Lee Sedol.</p>

<p><em><strong>[00:28:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He wants to see what Lee Sedol's going to think of that move because it was made while he was out on break. And Lee Sedol comes in and looks at the board. And here's what Lee Sedol said about it.</p>

<p><em><strong>[00:28:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He said, I thought AlphaGo was based on probability calculation and that it was merely a machine. But when I saw this move, I changed my mind. Surely AlphaGo is creative.</p>

<p><em><strong>[00:28:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This move was really creative and beautiful. So Lee Sedol sees that this move is not a mistake, that AlphaGo has made a move that humans wouldn't normally make. Human players do not make a move like this.</p>

<p><em><strong>[00:28:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They're built up heuristics and intuitions and such. You don't make a move like this. But when Lee Sedol saw the move, while everybody else was thinking a mistake, he suddenly realized, whoa, that was a smart move, even though it was one that was a very inhuman sort of move.</p>

<p><em><strong>[00:29:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's interesting to hear the commentators at this point. Let me see, they've got David Silver. He says, the professional commentators almost unanimously said that not a single human player would have chosen move 37.</p>

<p><em><strong>[00:29:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I actually had to poke around at AlphaGo. AlphaGo has, you can ask it questions. It's got analysis of its own moves and things like that.</p>

<p><em><strong>[00:29:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You can pull up the data and look at it. So I had to poke around at AlphaGo to see what AlphaGo thought. AlphaGo actually agreed with the assessment.</p>

<p><em><strong>[00:29:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> AlphaGo said that there was a one in 10,000 probability that move 37 would have been played by a human player. So it knew this was an extremely unlikely move. It went beyond its human guide, and it came up with something new and creative and different, is what David Silver said.</p>

<p><em><strong>[00:30:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And then one of the narrators in the show, he says, I am very much watching the game through the commentators. That's the way it works. So when they're confused, I'm certainly confused.</p>

<p><em><strong>[00:30:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> At the same time, I'm latching on to the fact that they are confused, right? This is, this, that is an interesting moment. When everyone is confused, who is not confused, right?</p>

<p><em><strong>[00:30:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Besides the machine. So Alpha Go with Move 37, it made a creative move that was outside of human knowledge. This is one of the things that was so interesting about it.</p>

<p><em><strong>[00:30:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And yet the world master, even if most of the commentators couldn't tell, immediately goes, wow, that's an amazingly beautiful move. And starts to realize, uh-oh, I'm not playing a normal computer Go playing program here. So that was one of the most exciting moments.</p>

<p><em><strong>[00:30:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That was early on in the game where it makes this move, kind of makes people start to realize this is no regular Go program. And Lee Sedol starts to get kind of worried, uh-oh, what's going on here? You know, this is something that can think differently than a human.</p>

<p><em><strong>[00:31:09]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's not like a human master, but it's a master all its own. He goes on to lose the first game. Of course, it's humiliating that he lost to this game, to this computer, but he's got a great attitude about it.</p>

<p><em><strong>[00:31:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And he's like, I'm going to come back tomorrow, and I'm going to win the next game. And he's kind of even excited, maybe, that he's found an opponent that's worthy of playing, right? It also leads to when AlphaGo has a victory, the media attention starts to get a lot more serious, because now it's actually beat the World Go Champion once.</p>

<p><em><strong>[00:31:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So that makes it way more interesting of an event. So the media scrutiny and attention starts to really heat up at this point in the story. So now I don't know, I can't remember the exact order of games that takes place.</p>

<p><em><strong>[00:32:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I believe it was AlphaGo. No, I do remember. AlphaGo goes on to beat Lee Sedol three times in a row.</p>

<p><em><strong>[00:32:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And at this point, Lee Sedol is actually lost, right? He's still got two more games to play. But AlphaGo is going to be the overall victor at this point because it's already won three out of five games.</p>

<p><em><strong>[00:32:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Lee Sedol has got a great attitude about this, though. His tone has changed, though. He's gone from, you know, I'm going to beat it every game to I'm going to beat it at least once.</p>

<p><em><strong>[00:32:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And this is what all the humans are cheering for now. There's so much tension over, can Lee Sedol beat Alpha Go at least once? The excitement is around that now.</p>

<p><em><strong>[00:32:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Alpha Go is no longer thought of as, oh, this little Go program. It's now the world champion, effectively, right? It's like, can we get a human to beat this unstoppable machine, Go playing machine?</p>

<p><em><strong>[00:32:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And a lot of the tension in the movie kind of shifts towards that at this point, where Lee Sedol is trying so hard to beat Alpha Go at least once. And this is where one of the... So, Move 37 is one of the big spoilers.</p>

<p><em><strong>[00:33:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This next one, the God Move, is probably the biggest spoiler. So that's why I hopefully watch the movie first. Lee Sedol comes back and beats Alpha Go on the fourth game.</p>

<p><em><strong>[00:33:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Everybody's celebrating and they're so excited, and it's like, yes, we humans, we're not out of the race, you know? There's a great deal of tension around that. But what's really interesting is how Lee Sedol beat Alpha Go in game four.</p>

<p><em><strong>[00:33:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They called the move that he used where he beat Alpha Go, they call it the God Move. I think it's also called Move 78 or something like that. I can't remember exactly what it was called.</p>

<p><em><strong>[00:33:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Lee Sedol, as he's playing Alpha Go, he starts to realize that it's teaching him ways to play Go that he had never thought of before because he's only ever played humans before, and that he's starting to think differently about the game of Go because Alpha Go treats the game of Go differently than a human would. In game five, we can explain better what I mean by that because it becomes super apparent how Alpha Go plays differently than a human during game five. But Lee Sedol is already starting to pick up on it, and he's starting to think of new sorts of strategies that he's never thought of before because of him playing against Alpha Go.</p>

<p><em><strong>[00:34:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So Move 37 in particular taught him something about the game he didn't know, and it caused him to then come up with the God Move so that he was able to beat Alpha Go. This is what I think is interesting, is how Alpha Go actually created a whole new play style that Lee Sedol then started to pick up on and started to use against it. And I think that's part of what makes game four so exciting is when he kind of figures out, ah, I get now how I can beat the machine.</p>

<p><em><strong>[00:34:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So here is what, here's the quote. It says, the lessons that Alpha Go is teaching us are going to influence how Go is played for the next thousand years. At the very end, the Go consultant, he says, where we look back and say, yeah, that was just like Move 37.</p>

<p><em><strong>[00:35:11]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Something beautiful occurred there, at least in a broad sense, Move 37 begat Move 78, begat a new attitude in Lee Sedol, a new way of seeing the game. He improved through this machine. His humanness was expanded by playing this inanimate creation.</p>

<p><em><strong>[00:35:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And that's actually true. From what I understand, world champion Go players have whole new play styles because of the advent of Alpha Go. That it introduced new play styles into the game that nobody had ever thought of before.</p>

<p><em><strong>[00:35:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And it changed the way humans play Go also. So then probably, so after Lee Sedol beats Alpha Go in game four, and everybody's really excited about this victory for the humans. Honestly, Lee Sedol might as well have won the entire tournament.</p>

<p><em><strong>[00:35:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The level of excitement of him beating Alpha Go once was just so through the roof. Everyone was so excited to see a human stomp the machine for a change. The fact that he had actually already lost three games almost didn't matter because at this point, everyone knew Alpha Go was nearly unstoppable.</p>

<p><em><strong>[00:36:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> That's kind of weird. The underdog suddenly won and he was no underdog, really.</p>

<p><em><strong>[00:36:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yes, he's the world champion that we're talking about. They go into game five and Lee Sedol is feeling somewhat confident that maybe he can beat Alpha Go again and get at least two victories against it. This leads to probably the most interesting aspect of the documentary.</p>

<p><em><strong>[00:36:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Even though the God move is super interesting and move 37 is super interesting, this last part is almost like a comedy. It's funny. So what happens is that early on in game five...</p>

<p><em><strong>[00:36:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Oh, by the way, one of the things that happened is when Lee Sedol made the God move, it made Alpha Go become a little confused. Just like we were talking about, the hallucination where it deludes itself. I don't think it necessarily played bad, but it could not quite figure out what to do from that point forward.</p>

<p><em><strong>[00:37:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And it was kind of obvious that it couldn't. He had figured out a super creative move that it hadn't foreseen, strangely as that may sound, that allowed him to go into a board position where it just couldn't figure out how to recover. It continued to play from that point forward, but it was just it was kind of obvious it was over.</p>

<p><em><strong>[00:37:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Just like when it made move 37, it became kind of obvious. Oh, once they started to realize what it had done, they started to realize, oh, the game's over. Alpha Go with that move, it had won the game.</p>

<p><em><strong>[00:37:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's just a matter of time now. And apparently that happens in Go. So in game five, in the final game, Alpha Go makes a move, and similar to move 37, everybody thinks it's a mistake.</p>

<p><em><strong>[00:37:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Only this one was a lot weirder. This one was… Move 37, at least Lee Sedol could see it was a beautiful move, right? The world champion.</p>

<p><em><strong>[00:38:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> In this case, everybody thought it was a mistake. So here's what the commentators are saying in the documentary. They say, is it fair to say that Alpha Go made a mistake?</p>

<p><em><strong>[00:38:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We might have another victory today for Lee Sedol. You weren't crazy about the timing of this move. The other guy says, yeah, I'm sort of thinking that maybe Alpha Go hasn't recovered from game four yet.</p>

<p><em><strong>[00:38:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yeah, they kind of chuckle. And then one of them says, I think it could be a kind of misreading. It says, there's no reason for why he's playing that move.</p>

<p><em><strong>[00:38:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> These are like the commentators who are watching the game. You've got these people who are commenting on the game. And in the background, the programmers have seen this move.</p>

<p><em><strong>[00:38:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And one of the programmers says, still deluded? Another one says, we don't know. Another one says, that is looking good for Lee Sedol.</p>

<p><em><strong>[00:38:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And they're like, yeah, it looks like Lee Sedol is going to win this game. And the programmer says, are we seeing another short circuit? So they check what AlphaGo thinks.</p>

<p><em><strong>[00:39:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And one of the programmers reports, AlphaGo is saying it's 91% certain now it's going to win. And the other programmer says, yeah, because it's incorrect again. And the consultant said, it's a bad move.</p>

<p><em><strong>[00:39:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> They all think that the machine is just messing it up.</p>

<p><em><strong>[00:39:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's right. It says, oh, maybe AlphaGo's weakness comes back. It's a bad move, says the AlphaGo consultant that trained it.</p>

<p><em><strong>[00:39:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It continues after this move, after it makes this bad move. And everybody thinks that's it. Lee Sedol has won the game.</p>

<p><em><strong>[00:39:31]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It continues to make bad moves throughout the rest of the game. So it really looks like it's just hallucinating and it's just doing these crazy moves, right? As the game moves on, the commentators suddenly change and they go, wait, I think AlphaGo's winning.</p>

<p><em><strong>[00:39:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And it turns out that all these crazy moves that it was making were actually really intelligent moves that no human could recognize as a good move. The reason why, and one of the things that the programmer says is the whole game, we thought that AlphaGo was wrong about the board position. So it analyzes this board position.</p>

<p><em><strong>[00:40:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They thought it was hallucinating. Oh, it thinks it's got a 91% chance of winning, but it's hallucinating. It says we were super worried that, oh, it's going to play garbage.</p>

<p><em><strong>[00:40:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's going to be like lose in a very embarrassing way. And this continued the whole game. As it turns out, none of us know go well enough to accurately judge what AlphaGo is doing.</p>

<p><em><strong>[00:40:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And the narrator says, we all say some of AlphaGo's moves are so weird and strange and maybe mistakes. But after a game is finished, we have to doubt ourselves our judgment. What AlphaGo was doing is its true play style came out during game five.</p>

<p><em><strong>[00:40:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And the easiest way to explain it is that throughout the history of the world, people had been using number of points as a proxy for chance of winning. So human players consistently, and this makes sense, right? I mean, like this is no big mystery here, consistently tried to go for as many points as they could possibly get on the grounds that that would increase their margin for the win, right?</p>

<p><em><strong>[00:41:17]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And increase their chances of winning. AlphaGo realized that wasn't the way it worked, that you won by winning by one point. So it played it entirely differently.</p>

<p><em><strong>[00:41:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It would almost throw moves away if it had to. Once it had consolidated what it was going to win with, its board position it was going to win with, it would simply defend that board position. And it wouldn't even try to take new territory after that.</p>

<p><em><strong>[00:41:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And it would make moves that seemed like it was just throwing stuff away. It was just waiting, biding its time so that it could now win the game. And that was why it seemed like it was making all these stupid moves.</p>

<p><em><strong>[00:41:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But it actually knew it had won the game. It had already figured out it had won the game. And it was just waiting for Lee to come along and [unclear] everybody else, all the humans, to come along and realize, oh, it's won.</p>

<p><em><strong>[00:42:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And until near the end of the game, none of the humans recognized that it had already won the game way back with its first crazy move. If you think about it from just the theory of reinforcement learning standpoint, what reinforcement learning does is it tries to learn this board position is closer to the goal of winning than this other board position. That's what it's trying to learn.</p>

<p><em><strong>[00:42:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Okay, that isn't the same thing as getting the most points. It just isn't. You can see how being defensive might be a better strategy in some circumstances than being offensive.</p>

<p><em><strong>[00:42:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But human players never realized that before. They had never understood that you only have to win by a point. So sometimes it's better to be defensive than it is to be offensive.</p>

<p><em><strong>[00:42:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Well, AlphaGo had figured this out. Okay, that was part of what its algorithm figures out as it tries to work out the probability of winning from each board position. And when it had realized this, that's how come it created this whole new play style, where humans had just never realized, oh, we've been using points as a proxy for chances of winning, when really we should be worrying about chances of winning directly, because that's a better thing for us to be worrying about.</p>

<p><em><strong>[00:43:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is what led AlphaGo to introduce a whole new play style, was this realization that humans were making a mistake. Thousands of years of Go players were making the same mistake, and AlphaGo wasn't. And this is why it invented this new play style.</p>

<p><em><strong>[00:43:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is why it seemed so inhumanly like it was making mistakes, but it was actually winning the game. This is one of the most interesting parts of the overall game, is just the fact that it had discovered this whole new play style through the way the machine learning algorithms work. And it had come up with this idea, I'm just going to try to win by one point.</p>

<p><em><strong>[00:43:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I'm not going to try to win by as many points as I can. Once I'm convinced I can win by one point, that's it. I'm going to just make sure I win by one point and I'm done.</p>

<p><em><strong>[00:44:08]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And that's why Alpha Go's play style seemed so random at times to a human who only is looking at shouldn't it be taking more board positions? Shouldn't we try to score more points? And it didn't care.</p>

<p><em><strong>[00:44:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That was one of the most exciting points of the story to me is when they started to realize that that every human on the planet did not understand this aspect of Go that Alpha Go understood. Maybe I can make it just an aside here for a second. We talked about in one of the episodes the pseudo-Deutsch theory of knowledge.</p>

<p><em><strong>[00:44:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This idea that no no AI algorithm, no algorithm that has ever been invented by humans has ever created knowledge before. Based on, typically, there's different ways that this is explained. It's explained as, well, because the human actually input all the knowledge.</p>

<p><em><strong>[00:45:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And we know the human does input most of the knowledge, so that's not entirely false. Based also on this idea that the knowledge somehow exists in the data, which to me sounds kind of inductive, but anyhow, that this is typically how it's said, is that the knowledge is in the data and it's just sort of reorganizing the knowledge into a useful format. But that the knowledge all came from the programmer.</p>

<p><em><strong>[00:45:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, I'm not saying AlphaGo refutes that, because to be honest, this is an irrefutable theory. And which is the problem? The theory can't be falsified in any way, which is why it's not a good explanation.</p>

<p><em><strong>[00:45:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But I think that AlphaGo playing Lee Sedol certainly shows how far you have to stretch this theory to take it seriously, to where not a single human understood what AlphaGo learned by playing itself or by using this algorithm. Not a single human had ever actually realized, oh, you shouldn't be using points as a proxy for chances of winning. You should be paying attention to winning by at least one point.</p>

<p><em><strong>[00:46:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The fact that AlphaGo really and truly did come up with a brand new creative move, move 37, you know, that it created a new play style, that all of these things, something so creative that the world champion was the only one who could recognize the beauty of the move, right? It required a world champion to understand how beautiful the move was because it was so creative. And it was something so far outside anything any human had done.</p>

<p><em><strong>[00:46:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It wasn't part of the data that was fed into it from human players because human players don't make this move. You could still, of course, say, oh, well, but the knowledge actually came from the algorithm and humans inserted that. You can always say that.</p>

<p><em><strong>[00:46:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's the problem with the pseudo-Deutsch theory of knowledge, is that it can be used for any circumstance like this. It can apply to any circumstance. It could never ever be falsified or refuted.</p>

<p><em><strong>[00:46:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But this should show just how far we're stretching to make the claim that machine learning creates no knowledge. I would also point out that we could use the pseudo-Deutsch theory of knowledge to claim that humans don't create knowledge. We could say, oh, all knowledge actually comes from biological evolution and all the knowledge that humans display actually comes from that knowledge created in their genes.</p>

<p><em><strong>[00:47:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And, I mean, of course, this is a ridiculous theory. I mean, of course, it's a ridiculous theory. And then humans just take observations and, you know, using induction, of course, take observations and the knowledge is already in the observations.</p>

<p><em><strong>[00:47:33]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And they use these algorithms that are built into their head by the knowledge that came from the genes. And so all the knowledge actually comes from the genes and humans generate no knowledge at all. And the pseudo-Deutsch Theory of Knowledge could be used to prove this also, which is the problem that it can, it over explains.</p>

<p><em><strong>[00:47:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It explains every single possible outcome. There's no outcome that exists that it can't explain. That's not what we want in a good theory.</p>

<p><em><strong>[00:48:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What we want in a good theory is one that can be falsified, that doesn't explain every single possible outcome. I do think, though, that Alpha Go is one of the strongest challenges, where intuitively you look at what happened. Go watch the movie, and then really try to hold on to the pseudo-Deutsch Theory of Knowledge.</p>

<p><em><strong>[00:48:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And I think you'll find that it's starting to crack around the edges for you, right? It just doesn't make sense that there's no knowledge creation that took place here, because the knowledge that it's showing, none of the humans knew about it. It didn't exist in any of the human heads.</p>

<p><em><strong>[00:48:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It was something, there was great knowledge that came from the humans, but it went beyond that. It went beyond that and generated some of its own knowledge that allowed it to be the world champion, that allowed it to be a new kind of play style that no human had seen before. This is how this documentary ties into some of our past podcasts and kind of the importance of really understanding that machine learning does have a form of knowledge creation that's involved with it, that AI is creating knowledge in a sort of narrow setting.</p>

<p><em><strong>[00:49:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, it doesn't create knowledge beyond its narrow setting. Alpha Go isn't going to suddenly learn how to make pancakes tomorrow. It's stuck to whatever its specific knowledge space is that it is trained to do.</p>

<p><em><strong>[00:49:22]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's narrow AI. It's no different than any other narrow AI in that regard. But within that narrow domain, machine learning comes up with things that no human being has seen before and that no new human being knows how to do.</p>

<p><em><strong>[00:49:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And that's really the point that I kind of wanted to make to particularly the Deutschian community that often has embraced the pseudo-Deutsch theory of knowledge. That this is something that needs a stronger look. This is something where we really need Popperians looking at it saying, okay, what's really going on with machine learning?</p>

<p><em><strong>[00:49:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What is it doing, you know, that it's able to come up with new things like this, totally creative new things like this that no human knew prior to this point? All right, off my soapbox on that one.</p>

<p><em><strong>[00:50:06]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> And so I'm curious because you said that this is a simplistic AI. Well, you know, I think it's a pretty big moment to see this machine have an intuitive leap where it does something that nobody had taught it and it figures out something we're not capable of figuring out or couldn't imagine our intuition hadn't come up with.</p>

<p><em><strong>[00:50:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Yes.</p>

<p><em><strong>[00:50:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> What does it mean?</p>

<p><em><strong>[00:50:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> And or maybe it doesn't mean anything, but, you know, looking at general artificial intelligence, do we see this as a leap forward on our understanding of machine learning or our understanding of our ability to create an artificial general intelligence?</p>

<p><em><strong>[00:50:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Very good question. There are in some ways, and we'll do a podcast on this at some point, in some ways, what you're asking is, what is it that we're missing about general intelligence? And we don't know what we're missing, right?</p>

<p><em><strong>[00:51:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I mean, like, it's a mystery what it is that we just we're not even asking the right questions at this point. When we try to come up with an intelligent machine to go play Alpha Go, we do it in a certain way. We figure out, OK, we're going to have this state space and we're going to have it, you know, we're going to use reinforcement learning and there's not even really an attempt to make it generalize, to do anything.</p>

<p><em><strong>[00:51:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Some people will tell you reinforcement learning is a general learning algorithm. That's only true if you don't take into consideration that a human has to go and put the state space, the world space. You have to go teach it what the world space is for each problem that you want to solve.</p>

<p><em><strong>[00:51:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It never learns to solve a different problem except for the one it's been programmed to solve. The rest of the algorithm generalizes. Think of it like a module you have to plug in.</p>

<p><em><strong>[00:52:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The reinforcement learning algorithm will learn any world space it's given, but you have to first give it to it. You have to first say, here's how to represent every single board position for AlphaGo. Here is how you understand what a reward is.</p>

<p><em><strong>[00:52:24]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It has to have some sort of input from the real world that's a reward that has to be explained to it. It has to be programmed, not explained, but programmed. And then from that point forward, it will then play billions of games with itself.</p>

<p><em><strong>[00:52:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> More games than any humans ever played. And it will figure out which board positions are the better ones. Once it knows which board positions are the better ones, then you think about it.</p>

<p><em><strong>[00:52:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> All it has to do is it has to try each move and say, is this move going to give me a better board position than this move or this move or this move? The knowledge gets caught up in its board evaluation algorithm. That's really where its true knowledge exists.</p>

<p><em><strong>[00:53:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It generates this board evaluation algorithm. It's the board evaluation algorithm that is the function that no human has ever seen before, that it has come up with. And its board evaluation algorithm was so good that looking ahead only one move, it could play at professional levels.</p>

<p><em><strong>[00:53:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you let it look ahead more than one move, then it was even better. And presumably, to play Lee Sedol, you're not looking ahead one move, right? You're looking ahead as many moves as you can.</p>

<p><em><strong>[00:53:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And then think about the fact that looking ahead, trying out moves, that itself is a form of knowledge creation, because you're trying out each of the moves, you're figuring out which move is the best one. And then you're using the board evaluation algorithm to tell you, as a proxy for, that's the best move. So the combination of those two algorithms generate knowledge.</p>

<p><em><strong>[00:53:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, here's the interesting thing. Think about, like, IBM playing Kasparov when Big Blue beat Kasparov. Big Blue did not create a new play style.</p>

<p><em><strong>[00:53:56]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Big Blue did not create brand new sorts of moves that no one had ever seen before. People said, wow, that was a creative move [wording unclear]. Now, there's a good reason why it didn't.</p>

<p><em><strong>[00:54:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's because all the knowledge was caught up in trying to look ahead as many moves as it could, which is just too limited to create a whole new creative play style. Alpha Go learned its creative play style through its board evaluation algorithm. In Big Blue, Deep Blue I mean, sorry, the board evaluation algorithm was done by a human.</p>

<p><em><strong>[00:54:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It was just human knowledge inserted into the computer program. With Alpha Go, that board evaluation algorithm was created by the learning algorithm, the reinforcement learning algorithm, combined with the deep learning algorithm. And so the net result was that it had come up with a new way to think about the game, and a new way to understand the game, and new creative sorts of moves.</p>

<p><em><strong>[00:54:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Okay, that's because the minimax algorithm is just too weak. Yes, it creates knowledge, but the knowledge it creates is very local. It just simply says, given this board position, what's my next best move looking out seven moves, or however moves it can figure out.</p>

<p><em><strong>[00:55:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's more complicated than that. It tries to cull out moves that aren't good, and it tries to look ahead 13 moves maybe, or 20 moves on the few most interesting possibilities, or something like that. With Alpha Go, it actually plays billions of games, and then it figures out, based on those games, which board positions are the most likely to lead to a win.</p>

<p><em><strong>[00:55:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's the way reinforcement learning works. That was why Alpha Go came up with new creative play styles, and beautiful new moves, and things like that. Whereas Deep Blue never did.</p>

<p><em><strong>[00:55:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It just simply played chess well. That is the difference between AI regular AI good old-fashioned AI and machine learning, where machine learning actually comes up with creative new algorithms. Functions, I should say.</p>

<p><em><strong>[00:55:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Functions and algorithms can be the same thing. It comes up with creative new functions that no human has ever seen before, whereas with AI it's just simply trying to find the best move out of the possible moves at this position. Does that make sense?</p>

<p><em><strong>[00:56:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Yeah, yeah, that totally makes sense. But what does it mean for the future of AGI?</p>

<p><em><strong>[00:56:12]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I think every approach to AI that we're currently doing is the wrong approach for AGI. They're all interesting in their own right. OK, I mean, the fact that we can make up these automated algorithms that can generate new sorts of creativity, right?</p>

<p><em><strong>[00:56:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> New sorts of creative moves, things like that, within narrow domains. That's not a bad thing. When we actually create A.G.I.s, they're going to be a lot like us.</p>

<p><em><strong>[00:56:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They're going to be people. We aren't going to want to enslave them. I mean, that would be bad to enslave a person to then be a chess player only or something like that, right?</p>

<p><em><strong>[00:56:47]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you want to make a really good chess player, what you really want is you want a narrow AI to do it. And that's true of running manufacturing plants. I mean, most of what we're going to want to do through automation is going to be narrow AI forever.</p>

<p><em><strong>[00:57:01]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We're never going to use AGI to automate things. That wouldn't make sense because it's going to be a creative individual. So both fields need to be researched for different purposes.</p>

<p><em><strong>[00:57:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Both fields are types of knowledge creation. So there is an overlap between them. They're both different subsets of a single thing, which is knowledge creation.</p>

<p><em><strong>[00:57:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But they're very different sorts of knowledge creation. Now, why are they different? That's what we're trying to figure out, right?</p>

<p><em><strong>[00:57:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> One of the things that they talk about is there's a gentleman. He researches what is called the problem of open-endedness. Let me get his name here.</p>

<p><em><strong>[00:57:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Oh yes, Kenneth O. Stanley. And you can go look him up on YouTube, and he'll talk about the problem of open-endedness.</p>

<p><em><strong>[00:57:48]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, if you think about knowledge creation, everything that humans have created that creates knowledge, so basically AI algorithms and machine learning algorithms, they're all narrow knowledge creation. They create knowledge in some little tiny domain that we understand well enough to explain, and then it goes and it tries variants, and it discovers things by doing that. That's different, though, than the other two kinds of knowledge creation that David Deutsch mostly talks about, which is Neo-Darwinian evolution and human knowledge creation.</p>

<p><em><strong>[00:58:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now, both of those are, in some sense, open-ended. Think about, like, you know, Darwinian evolution and how it creates all sorts of different species, right? It's kind of, there's not, it's not limited.</p>

<p><em><strong>[00:58:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's not going to try to create the best mouse, and that's it, right? It will discover all sorts of new species and new creative ways to live in niches that weren't inhabited before. And that's what Neo-Darwinian evolution does.</p>

<p><em><strong>[00:58:52]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's got this open-endedness to it. And we don't know how to program that. Never mind AGI yet.</p>

<p><em><strong>[00:58:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's a related but different problem. We can't even program Darwinian evolution as an algorithm. Now, this is, this is what Leslie Valiant, I think I mentioned him in past episodes.</p>

<p><em><strong>[00:59:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is one of his areas of research. He's a very famous guy in artificial intelligence and machine learning, and he's written a number of books. And he points out that we think we understand Darwinian evolution and we don't.</p>

<p><em><strong>[00:59:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We don't really know what the algorithm is that creates the sort of effects that you see in Darwinian evolution. We try to, we have something called genetic programming, which is based on Darwin's theory of evolution. It does crossover, it does mutation, it does mating, it creates these population pools of replicators.</p>

<p><em><strong>[00:59:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It does everything that we think we understand about Darwinian evolution. And the end result is narrow AI, just like every other type of AI we've built. Even though it's got, in theory, because it uses a programming language, in theory, it's got an open-ended problem search space.</p>

<p><em><strong>[01:00:05]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It never uses it. It always just explores a little tiny part of that space that's super narrow, and it produces results nearly identical to any other form of narrow AI. It's just not a super creative algorithm, and it's certainly not open-ended.</p>

<p><em><strong>[01:00:25]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> In fact, it converges. The way they set it up, usually they're trying to have it solve a single problem, so it kind of makes sense that it converges. It's part of just how it's designed.</p>

<p><em><strong>[01:00:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But then you could say, oh, well, then don't have it converge to one thing. Let it just try to discover random solutions. I can say that, but I don't know what that means.</p>

<p><em><strong>[01:00:44]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I don't even understand what I just said well enough to go make an algorithm out of it. And this is one of the big secrets, and I brought this up in our computational theory episodes, is that human beings understand things through algorithms. And David Deutsch just put some warnings on that.</p>

<p><em><strong>[01:01:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> He said, well, maybe we shouldn't... It's been his experience that it's a mistake to try to jump to algorithms immediately, that we should start by explaining things without algorithms. Well, when you think about Darwinian evolution, we've gotten quite far with an explanation of Darwinian evolution that can't be fully algorithmicized yet.</p>

<p><em><strong>[01:01:21]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But insofar as we can't algorithmicize it, it shows that we don't understand parts of it. And maybe we think we do. I've had people where I've talked to them and I've said, yeah, we don't fully understand Darwinian evolution.</p>

<p><em><strong>[01:01:32]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They say, oh, yeah, we do. I've had people argue with me over this. But the simple truth is we don't.</p>

<p><em><strong>[01:01:37]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If we actually understood it completely, we could make open-ended evolution of virtual species. And believe me, they've tried to do this, right? They've had a certain level of success, but it always seems to kind of top off at some point, and it just stops creating new things at some point.</p>

<p><em><strong>[01:01:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's because we're missing something. There's something about the theory of evolution that we don't understand, and we don't even understand what it is that we don't understand. So it's hard to even ask the right question.</p>

<p><em><strong>[01:02:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And that's what Kenneth Stanley is trying to study is with the problem of open-endedness. One of the things that he's researched is trying to come up with creative search. So instead of trying to search for just a solution to the problem, so let's say you have a virtual robot that you're trying to teach to walk.</p>

<p><em><strong>[01:02:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Instead of just trying to directly give it rewards through reinforcement learning for successfully walking, he basically just tells it any new state you haven't reached before you get a reward for, and it will learn to walk faster than if you directly try to teach it to walk by doing that. And so that's like his approach to try to understand the problem of open-endedness. And he's got a number of other really interesting experiments that he's trying to do using software to explore the problem of open-endedness.</p>

<p><em><strong>[01:02:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So that's one problem that we know we don't understand correctly. Valiant's pointed out that we don't even understand a tractable version of Darwinian evolution, that the versions that we program would never tractably run, even if they could solve the problem of open-endedness, to be able to create species in a few billion years like the world was able to do so. So we're missing something there, and there's some interesting ideas there that are worthy of some research.</p>

<p><em><strong>[01:03:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> There was that paper that Thatchaphol in one of our episodes talked about, where someone tried to take Leslie Valiant's algorithmic evolutionary algorithm that he had proposed that didn't really work very well, and he got it to solve the bitwise problem—which is something that evolutionary algorithms can't solve—by introducing the idea of ecology into the mix. So maybe that's one of the things we're missing, right? Maybe we're so narrowly focusing on variation and selection that we're missing the fact that there's these aspects of ecology that are also knowledge creating, that have their own variation and selection, that allow it to be able to solve problems that it can't currently solve, that might lead us to an understanding of open-endedness.</p>

<p><em><strong>[01:04:15]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Beyond that, there's an additional problem for AGI, which is we talk about like science's explanations, right? And this is something we've talked about with our epistemology. And we say, oh yeah, science is about explaining things.</p>

<p><em><strong>[01:04:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> This is one of the things that like the instrumentalists have missed, which is why they misunderstand science, that science is really about finding the best explanation. It's about trying to creatively come up with a conjecture and then criticize that conjecture. And then the one that survives, that's the best one.</p>

<p><em><strong>[01:04:45]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's the one that has the most verisimilitude. It's the truest one. That all makes sense.</p>

<p><em><strong>[01:04:51]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But can you explain to me algorithmically what an explanation is? Because I don't think anybody can right now.</p>

<p><em><strong>[01:04:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> I sure can't.</p>

<p><em><strong>[01:04:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Right.</p>

<p><em><strong>[01:04:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Give me a minute.</p>

<p><em><strong>[01:05:02]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We have an upcoming episode where I'm going to talk about causal inference. One of the things that I found exciting about causal inference is that it tried to create a mathematical, computational graph model of what an explanation is.</p>

<p><em><strong>[01:05:18]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It's too primitive, though, right? I don't see how it's going to successfully create the kind of explanations that exist in science. It's these really primitive explanations.</p>

<p><em><strong>[01:05:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> There's something called explanation-based learning, which does the same thing, and it uses logic to be able to create its explanations, propositional logic or first order logic. And that kind of makes sense. Popper, his epistemology, if you go read his book, he's got several things that we call books, most of them are just collections of things that he wrote.</p>

<p><em><strong>[01:05:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Sometimes he has one actual book called The Logic of Scientific Discovery. His epistemology is based entirely on propositional and first order logic. And so it kind of makes sense that an explanation, he models explanations as logic statements.</p>

<p><em><strong>[01:06:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Now David Deutsch has criticized that a little and said that that's only an approximation of an explanation. Well, what does that mean? If logic statements are only an approximation of an explanation, then what is an explanation?</p>

<p><em><strong>[01:06:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> How would you model that inside of a computer? Well, nobody knows. We've got a couple of different paths, explanation-based learning, causal inference, where we're trying to come up with something.</p>

<p><em><strong>[01:06:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Right now they seem kind of primitive. Is that possibly one of the things that we're missing for AGI? Maybe.</p>

<p><em><strong>[01:06:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I don't know. Another one is how do we make conjectures? What's the conjecture engine?</p>

<p><em><strong>[01:06:41]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Popper basically treats conjectures as a mystery. He basically says you just creatively come up with your best conjecture, and then from there you criticize it. His epistemology doesn't explain how to come up with a conjecture.</p>

<p><em><strong>[01:06:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It explains what to do once you have one. Well, how do you actually come up with one? Well, if I knew that, I probably would have solved the problem of AGI.</p>

<p><em><strong>[01:07:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That's something missing. And again, what does it even mean to creatively come up with a conjecture? Could you give me an algorithm that explains what those words mean?</p>

<p><em><strong>[01:07:13]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> They're actually very vague. You may not think of them as vague, but the moment you try to put them into an algorithm, you'll start to realize just how vague they actually are. And this is why I really favor computational theory approach to things.</p>

<p><em><strong>[01:07:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> There is value in creating theories that we don't know how to turn into an algorithm. And that usually is the first step. You have to first create the theory at the level of linguistics.</p>

<p><em><strong>[01:07:38]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But there's these vaguenesses that exist in such a theory that you may not even be aware are there until you try to put them into an algorithm. And the moment you try to put them into an algorithm, you start to realize, oh, wow, there are massive gaps in my knowledge that I didn't even realize I have until I tried to get specific and put it into an algorithm. And I think that all of these are what we're missing, right?</p>

<p><em><strong>[01:08:04]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We need to explore out a lot of these ideas better, and then we need to make a failed attempt to put them into algorithms and then figure out what is it that I'm missing by failing. This is something Popper brings up over and over again. He says the way you actually solve a problem is by trying to solve it and failing, and that helps you understand the problem.</p>

<p><em><strong>[01:08:23]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And once you understand the problem really well, that's when you actually are in a position to try to actually solve it for real. So in other words, you have to go try to fail to solve a problem, to educate yourself on the problem to the point where you have any chance of solving the problem. If that makes any sense.</p>

<p><em><strong>[01:08:40]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Not only does it make sense, but I, like, we could almost have an episode just on that. That's a place I've been kind of thinking deeply about. You know, just in my job, you see a lot of organizations trying to become more lean or more agile.</p>

<p><em><strong>[01:08:57]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> But ultimately, the goal isn't to—isn't to be agile. It's to figure out how to effectively fail with within organizational constraints, you know, and traditionally, businesses don't like the concept of failure. And so part of what I think is you see kind of happening in a lot of, especially in software development is how can we organizationally get to a place where we are more comfortable with the concept of failing because it's the only actual way we can learn.</p>

<p><em><strong>[01:09:30]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> So I'm just very interested in that in that particular concept right now.</p>

<p><em><strong>[01:09:34]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> That is something that follows directly from Popper's epistemology. What you really want is you want to be able to fail and not cause problems because then you can learn faster. There are some people who consider themselves to be critical rationalists who really just don't get this fact.</p>

<p><em><strong>[01:09:49]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> And they've got these concepts of you shouldn't overreach because then you'll make lots of mistakes and that's failure and failure is bad. It's like, wow, you call yourself a critical rationalist. How have you so severely misunderstood the epistemology?</p>

<p><em><strong>[01:10:03]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> What you really want is sometimes failure is bad, right? That's why people hate it. But what you really want is you want a situation where failure is not so bad and that you can figure out how to let the failures through, not cause your whole organization to fall apart over it.</p>

<p><em><strong>[01:10:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You know, someone who's really good at this is Amazon. Amazon has an enormous number of failures.</p>

<p><em><strong>[01:10:26]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Sure.</p>

<p><em><strong>[01:10:27]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> And they're very, very good at failing, learning from the failure and then moving on and using that failure as a jumping point toward a future success.</p>

<p><em><strong>[01:10:39]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Yes. And if you just think about it from just a variation selection, which that's knowledge creation, right? Standpoint, if you go out and you try, you know, 20 different ideas and 19 of them fail, but one of them is a massive hit, it pays for the failures.</p>

<p><em><strong>[01:11:00]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Right. I mean, like you're going to find massive hits by trying things and failing. And then by chance, you get one that's really good.</p>

<p><em><strong>[01:11:10]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Tracy:</span></strong> Oddly enough, it seems like failure is actually just confirmational. That's it.</p>

<p><em><strong>[01:11:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We're probably coming up on time here. So any other final thoughts on Alpha Go the Movie?</p>

<p><em><strong>[01:11:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#9333ea">Cameo:</span></strong> Well, hopefully nobody ruined it for themselves, but even if they hadn't watched it, I still recommend going and watching it because I think the impact of seeing like the way Lee Sedol is responding to this and the way that everybody's constantly kind of surprised by what's going on is hard to convey here. And it's just a really enjoyable show. It just is really enjoyable.</p>

<p><em><strong>[01:11:46]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I agree. Me quoting them just doesn't do it justice. You have to kind of live it, and that's what the movie lets you do.</p>

<p><em><strong>[01:11:53]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> You have to really see, wow, something really creative is going on here, right? The algorithm is doing something creative, and it's coming up with things that the humans just don't get. Only within its own little narrow area.</p>

<p><em><strong>[01:12:07]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> We're still talking narrow AI for sure, right? This is not a conscious AGI we're talking about at all. But there's something just interesting in and of itself about machine learning.</p>

<p><em><strong>[01:12:19]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Machine learning is an interesting topic all its own, regardless of whether it's a path to AGI or not, which it's not, not the current form anyhow. Yeah, definitely would encourage people to take a look at Alpha Go the Movie and experience this for themselves, really be there and watch what's happening and how it unfolds and how it feels to the people that are involved. All right.</p>

<p><em><strong>[01:12:42]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Well, thank you guys. This has been entertaining to talk about this movie with you. It's one of my favorite movies, honestly.</p>

<p><em><strong>[01:12:50]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I recommend it to anybody. So thanks, everybody.</p>

<p><em><strong>[01:12:54]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Thank you.</p>

<p><em><strong>[01:12:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#6b7280">Unknown speaker:</span></strong> Thanks, everyone.</p>

<p><em><strong>[01:12:59]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> After I finished recording this episode on AlphaGo, David Deutsch posted a video on his YouTube channel called Popper's Problem Oriented Epistemology with David Deutsch and Eli Tyre. And in the video, he actually talks with Eli about AlphaGo briefly. Eli makes the claim that AlphaGo creates knowledge, and David Deutsch admits that maybe it does, although he still seems somewhat skeptical of that possibility.</p>

<p><em><strong>[01:13:20]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> But he mentions, I think this is correct, that if it does create knowledge, it creates it in the same way that biological evolution does, not through explanations. I don't think there's any doubt about that at all. The type of knowledge that is created by a program like AlphaGo is not explanatory knowledge.</p>

<p><em><strong>[01:13:36]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> So I was really glad to see David Deutsch finally address AlphaGo directly like this, and I was glad to see that he's at least open to the possibility that AlphaGo does create knowledge. This is a good example of what I was trying to get at within the podcast, that AlphaGo does present a problem for the pseudo-Deutsch theory of knowledge, and that was really my point here. Since the pseudo-Deutsch theory of knowledge is irrefutable, it could be applied to AlphaGo very easily.</p>

<p><em><strong>[01:13:58]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> It could be applied to literally anything very easily. But I think emotionally it becomes far more difficult to apply it to something like AlphaGo, where it's clearly come up with this entirely new creative play style that's never been seen before in the history of the world. In any case, this is the point I was really trying to make.</p>

<p><em><strong>[01:14:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> I really wasn't trying to go beyond that point, but I was glad to see that David Deutsch could see that there was a problem here and is starting to open his mind to the possibility that machine learning algorithms do create knowledge.</p>

<p><em><strong>[01:14:28]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The Theory of Anything podcast could use your help. We have a small but loyal audience, and we'd like to get the word out about the podcast to others so others can enjoy it as well. To the best of our knowledge, we're the only podcast that covers all four strands of David Deutsch's philosophy as well as other interesting subjects.</p>

<p><em><strong>[01:14:43]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you're enjoying this podcast, please give us a 5-star rating on Apple Podcasts. This can usually be done right inside your podcast player. Or you can Google The Theory of Anything Podcast Apple or something like that.</p>

<p><em><strong>[01:14:55]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> Some players have their own rating system, and giving us a 5-star rating on any rating system would be helpful. If you enjoy a particular episode, please consider tweeting about us or linking to us on Facebook or other social media to help get the word out. If you are interested in financially supporting the podcast, we have two ways to do that.</p>

<p><em><strong>[01:15:14]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> The first is via our podcast host site, Anchor. Just go to anchor.fm slash four dash strands, F-O-U-R dash S-T-R-A-N-D-S. There's a support button available that allows you to do reoccurring donations.</p>

<p><em><strong>[01:15:29]</strong></em>&nbsp;&nbsp;<strong><span style="color:#dc2626">Bruce:</span></strong> If you want to make a one-time donation, go to our blog, which is fourstrands.org. There is a donation button there that uses PayPal. Thank you.</p>
