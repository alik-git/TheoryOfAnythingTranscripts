# Remove legacy opening notices; the shared footer carries the current notice.
Jekyll::Hooks.register :pages, :pre_render do |page|
  next unless page.path.start_with?("episodes/")

  page.content = page.content.lines.reject do |line|
    line.start_with?("> Unofficial, reviewed transcript.") ||
      line.start_with?("Edited from the preserved Apple transcript using a separate full speech-recognition pass")
  end.join
end
