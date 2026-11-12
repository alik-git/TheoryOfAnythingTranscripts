# Keep the requested opening line consistent across existing episode pages.
Jekyll::Hooks.register :pages, :pre_render do |page|
  next unless page.path.start_with?("episodes/")

  notice = "These transcripts were generated with help from AI. They may contain mistakes; please refer to the podcast to clarify."
  page.content = page.content.lines.reject do |line|
    line.start_with?("> Unofficial, reviewed transcript.") ||
      line.start_with?("Edited from the preserved Apple transcript using a separate full speech-recognition pass") ||
      line.strip == notice
  end.join
  page.content = page.content.sub("## Transcript\n", "#{notice}\n\n## Transcript\n")
end
